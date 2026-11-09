"""DreamerV3 agent: world model + actor-critic learned in imagination (PyTorch).

Losses follow the paper (Hafner et al., DreamerV3):
  world model  L = L_pred + 0.5 L_dyn + 0.1 L_rep, free bits 1 nat, L_pred = image MSE (sum over pixels) + symlog MSE
               (vector) + two-hot reward + Bernoulli continue
  critic       two-hot on lambda-returns (lambda 0.95, gamma 0.997) + regulariser towards an EMA slow critic (0.02)
  actor        REINFORCE with advantages scaled by max(1, EMA(P95 - P5) of returns), entropy 3e-4
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nets import RSSM, SIZES, Actor, Decoder, Encoder, Head, TwoHot, symlog


class WorldModel(nn.Module):
    def __init__(self, img_ch, vec_dim, act_dim, size="size200m", img=64):
        super().__init__()
        deter, hidden, classes, depth, units = SIZES[size]
        self.enc = Encoder(img_ch, vec_dim, depth, units, img=img)
        self.rssm = RSSM(act_dim, self.enc.out_dim, deter, hidden, classes=classes)
        self.feat_dim = deter + 32 * classes
        self.dec = Decoder(self.feat_dim, img_ch, vec_dim, depth, units, img=img)
        self.rew = Head(self.feat_dim, units, 1, 255)
        self.con = Head(self.feat_dim, units, 1, 1)
        self.units = units

    @staticmethod
    def feat(s):
        return torch.cat([s["deter"], s["stoch"].flatten(-2)], -1)

    vec_scale = 1.0  # weight of the vector reconstruction loss (the image loss is a sum over 64*64*4 values)

    def loss(self, batch, twohot, state=None):
        img, vec, act = batch["image"], batch["vector"], batch["action"]
        b, t = img.shape[:2]
        tokens = self.enc(img.flatten(0, 1), vec.flatten(0, 1)).reshape(b, t, -1)
        out, last = self.rssm.observe(tokens, act, batch["is_first"], state)
        feat = self.feat(out)
        ff = feat.flatten(0, 1)
        img_pred, vec_pred = self.dec(ff)
        target = img.flatten(0, 1).permute(0, 3, 1, 2).float() / 255.0 - 0.5
        l_img = (img_pred.float() - target).pow(2).sum((1, 2, 3)).reshape(b, t)
        l_vec = (vec_pred.float() - symlog(vec.flatten(0, 1))).pow(2).sum(-1).reshape(b, t)
        l_rew = twohot.loss(self.rew(ff), batch["reward"].flatten(0, 1)).reshape(b, t)
        cont = 1.0 - batch["is_terminal"].float()
        l_con = F.binary_cross_entropy_with_logits(self.con(ff)[..., 0].float(), cont.flatten(0, 1),
                                                   reduction="none").reshape(b, t)
        dyn, rep = self.rssm.kl(out["post"], out["prior"])
        l_dyn, l_rep = torch.clamp(dyn, min=1.0), torch.clamp(rep, min=1.0)
        total = (l_img + self.vec_scale * l_vec + l_rew + l_con + 0.5 * l_dyn + 0.1 * l_rep).mean()
        metrics = {"loss": total, "img": l_img.mean(), "vec": l_vec.mean(), "rew": l_rew.mean(), "con": l_con.mean(),
                   "kl": dyn.mean(), "img_mse_px": l_img.mean() / target[0].numel()}
        return total, metrics, out, last


class ActorCritic(nn.Module):
    def __init__(self, feat_dim, act_dim, units, horizon=15, gamma=0.997, lam=0.95, ent=3e-4):
        super().__init__()
        self.actor = Actor(feat_dim, act_dim, units)
        self.critic = Head(feat_dim, units, 3, 255)
        self.slow = copy.deepcopy(self.critic).requires_grad_(False)
        self.horizon, self.gamma, self.lam, self.ent = horizon, gamma, lam, ent
        self.register_buffer("ret_lo", torch.zeros(()))
        self.register_buffer("ret_hi", torch.zeros(()))

    def update_slow(self, mix=0.02):
        with torch.no_grad():
            for s, p in zip(self.slow.parameters(), self.critic.parameters()):
                s.lerp_(p, mix)

    def imagine(self, wm, start):
        """start: dict deter (N, D), stoch (N, S, C). Returns feats (H+1, N, F), actions (H, N, A), logprob, entropy."""
        state = {k: v.detach() for k, v in start.items()}
        feats, acts, logps, ents = [wm.feat(state)], [], [], []
        for _ in range(self.horizon):
            dist = self.actor(feats[-1].detach())
            a = dist.sample().clamp(-1, 1)
            logps.append(dist.log_prob(a).sum(-1))
            ents.append(dist.entropy().sum(-1))
            acts.append(a)
            with torch.no_grad():
                state = wm.rssm.img_step(state, a.to(state["deter"].dtype))
            feats.append(wm.feat(state))
        return torch.stack(feats), torch.stack(acts), torch.stack(logps), torch.stack(ents)

    def loss(self, wm, start, start_cont, twohot):
        feats, acts, logp, ent = self.imagine(wm, start)
        with torch.no_grad():
            rew = twohot.mean(wm.rew(feats[1:]))  # (H, N)
            con = torch.sigmoid(wm.con(feats[1:])[..., 0].float())
            con = torch.cat([start_cont[None].float(), con], 0)  # (H+1, N)
            val = twohot.mean(self.critic(feats))  # (H+1, N)
            disc = self.gamma * con
            rets = [val[-1]]
            for k in reversed(range(self.horizon)):
                rets.append(rew[k] + disc[k + 1] * ((1 - self.lam) * val[k + 1] + self.lam * rets[-1]))
            ret = torch.stack(rets[::-1][:-1])  # (H, N) lambda-returns for states 0..H-1
            weight = torch.cumprod(torch.cat([torch.ones_like(disc[:1]), disc[:-1]], 0), 0)[:-1]
            lo, hi = torch.quantile(ret.flatten(), 0.05), torch.quantile(ret.flatten(), 0.95)
            self.ret_lo.lerp_(lo, 0.01)
            self.ret_hi.lerp_(hi, 0.01)
            scale = torch.clamp(self.ret_hi - self.ret_lo, min=1.0)
            adv = (ret - val[:-1]) / scale
        actor_loss = -(weight * (logp * adv + self.ent * ent)).mean()
        logits = self.critic(feats[:-1].detach())
        with torch.no_grad():
            slow = twohot.mean(self.slow(feats[:-1]))
        critic_loss = (weight * (twohot.loss(logits, ret) + twohot.loss(logits, slow))).mean()
        metrics = {"actor": actor_loss, "critic": critic_loss, "ret": ret.mean(), "val": val.mean(),
                   "ent": ent.mean(), "ret_scale": scale, "imag_rew": rew.mean(), "imag_con": con[1:].mean(),
                   "act_abs": acts.abs().mean()}
        return actor_loss, critic_loss, metrics
