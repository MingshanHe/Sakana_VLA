"""DreamerV3 networks in PyTorch, after the official implementation (danijar/dreamerv3, 2024 version).

- RSSM with a block-diagonal GRU-style core (deter split into `blocks` groups), 32 x `classes` categorical latents
  with 1% unimix, straight-through sampling.
- CNN encoder (conv k5 + 2x2 max-pool + RMSNorm + SiLU per stage, down to 4x4) and an MLP for symlog vectors.
- Mirror-image decoder (nearest upsample + conv), symlog MSE for vectors, MSE for the image.
- Reward / value heads: symexp two-hot over 255 bins in [-20, 20] (symlog space); continue head: Bernoulli.
- Actor for continuous actions: tanh-bounded mean, std in [0.1, 1].
Sizes follow the official presets; `size200m` is the default configuration of DreamerV3.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

SIZES = {  # deter, hidden, classes, cnn depth, units (stoch = 32, blocks = 8)
    "size12m": (2048, 256, 16, 16, 256),
    "size25m": (3072, 384, 24, 24, 384),
    "size50m": (4096, 512, 32, 32, 512),
    "size100m": (6144, 768, 48, 48, 768),
    "size200m": (8192, 1024, 64, 64, 1024),
    "size400m": (12288, 1536, 96, 96, 1536),
}


def symlog(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x):
    return torch.sign(x) * torch.expm1(torch.abs(x))


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-4):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.scale).to(dtype)


class ChanRMSNorm(RMSNorm):
    def forward(self, x):  # (B, C, H, W)
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


def dense(i, o):
    return nn.Sequential(nn.Linear(i, o, bias=False), RMSNorm(o), nn.SiLU())


def mlp(i, units, layers):
    mods = []
    for _ in range(layers):
        mods.append(dense(i, units))
        i = units
    return nn.Sequential(*mods)


class BlockLinear(nn.Module):
    def __init__(self, i, o, blocks):
        super().__init__()
        assert i % blocks == 0 and o % blocks == 0
        self.blocks = blocks
        self.weight = nn.Parameter(torch.empty(blocks, i // blocks, o // blocks))
        self.bias = nn.Parameter(torch.zeros(o))
        nn.init.trunc_normal_(self.weight, std=1.0 / math.sqrt(i // blocks))

    def forward(self, x):  # (..., i)
        lead = x.shape[:-1]
        x = x.reshape(*lead, self.blocks, -1)
        x = torch.einsum("...gi,gio->...go", x, self.weight.to(x.dtype))
        return x.reshape(*lead, -1) + self.bias.to(x.dtype)


class TwoHot:
    def __init__(self, bins=255, low=-20.0, high=20.0, device="cuda"):
        self.bins = torch.linspace(low, high, bins, device=device)

    def mean(self, logits):
        p = torch.softmax(logits.float(), -1)
        return symexp((p * self.bins).sum(-1))

    def loss(self, logits, target):
        """-log p(target) under two-hot encoding of symlog(target)."""
        x = symlog(target.float()).clamp(self.bins[0], self.bins[-1])
        idx = torch.bucketize(x, self.bins).clamp(1, len(self.bins) - 1)
        lo, hi = self.bins[idx - 1], self.bins[idx]
        w_hi = ((x - lo) / (hi - lo)).clamp(0, 1)
        logp = torch.log_softmax(logits.float(), -1)
        return -(logp.gather(-1, (idx - 1)[..., None])[..., 0] * (1 - w_hi) + logp.gather(-1, idx[..., None])[..., 0] * w_hi)


class RSSM(nn.Module):
    def __init__(self, act_dim, token_dim, deter, hidden, stoch=32, classes=64, blocks=8, unimix=0.01,
                 img_layers=2, obs_layers=1, dyn_layers=1):
        super().__init__()
        self.deter, self.stoch, self.classes, self.blocks, self.unimix = deter, stoch, classes, blocks, unimix
        self.x0 = dense(stoch * classes, hidden)
        self.x1 = dense(deter, hidden)
        self.x2 = dense(act_dim, hidden)
        gi = deter // blocks + 3 * hidden
        self.dyn = nn.ModuleList()
        for _ in range(dyn_layers):
            self.dyn.append(nn.ModuleDict({"lin": BlockLinear(blocks * gi if not len(self.dyn) else deter, deter, blocks),
                                           "norm": RMSNorm(deter)}))
        self.gates = BlockLinear(deter, 3 * deter, blocks)
        self.img = mlp(deter, hidden, img_layers)
        self.img_out = nn.Linear(hidden, stoch * classes)
        self.obs = mlp(deter + token_dim, hidden, obs_layers)
        self.obs_out = nn.Linear(hidden, stoch * classes)

    def initial(self, b, device, dtype=torch.float32):
        return {"deter": torch.zeros(b, self.deter, device=device, dtype=dtype),
                "stoch": torch.zeros(b, self.stoch, self.classes, device=device, dtype=dtype)}

    def _logits(self, x):
        return x.reshape(*x.shape[:-1], self.stoch, self.classes)

    def _dist_sample(self, logits):
        probs = torch.softmax(logits.float(), -1)
        probs = (1 - self.unimix) * probs + self.unimix / self.classes
        u = torch.rand_like(probs).clamp(1e-20, 1.0 - 1e-7)
        gumbel = -torch.log(-torch.log(u))  # Gumbel-max sampling (compile-friendly multinomial)
        idx = (torch.log(probs) + gumbel).argmax(-1)
        onehot = F.one_hot(idx, self.classes).to(probs.dtype)
        return (onehot + probs - probs.detach()), probs

    def core(self, deter, stoch, action):
        action = action / torch.clamp(action.abs(), min=1.0).detach()
        x = torch.cat([self.x0(stoch.flatten(-2)), self.x1(deter), self.x2(action)], -1)
        g = self.blocks
        x = x[..., None, :].expand(*x.shape[:-1], g, x.shape[-1])
        x = torch.cat([deter.reshape(*deter.shape[:-1], g, -1), x], -1).flatten(-2)
        for layer in self.dyn:
            x = F.silu(layer["norm"](layer["lin"](x)))
        x = self.gates(x)
        x = x.reshape(*x.shape[:-1], g, 3, -1)
        reset, cand, update = [x[..., k, :].flatten(-2) for k in range(3)]
        reset = torch.sigmoid(reset)
        cand = torch.tanh(reset * cand)
        update = torch.sigmoid(update - 1)
        return update * cand + (1 - update) * deter

    def prior_logits(self, deter):
        return self._logits(self.img_out(self.img(deter)))

    def post_logits(self, deter, tokens):
        return self._logits(self.obs_out(self.obs(torch.cat([deter, tokens], -1))))

    def observe(self, tokens, actions, is_first, state=None):
        """tokens (B, T, D), actions (B, T, A) taken at each step, is_first (B, T).
        h_t = core(h_{t-1}, z_{t-1}, a_{t-1}); reset to zeros where is_first."""
        b, t = tokens.shape[:2]
        if state is None:
            state = self.initial(b, tokens.device, tokens.dtype)
            prev_a = torch.zeros_like(actions[:, 0])
        else:
            state, prev_a = state
        deters, stochs, posts, priors = [], [], [], []
        step = self._obs_step_compiled or self._obs_step
        for k in range(t):
            deter, stoch, post, prior = step(state["deter"], state["stoch"], prev_a, tokens[:, k], is_first[:, k])
            state = {"deter": deter, "stoch": stoch}
            prev_a = actions[:, k]
            deters.append(deter)
            stochs.append(stoch)
            posts.append(post)
            priors.append(prior)
        out = {"deter": torch.stack(deters, 1), "stoch": torch.stack(stochs, 1), "post": torch.stack(posts, 1),
               "prior": torch.stack(priors, 1)}
        return out, (state, prev_a)

    _obs_step_compiled = None
    _img_step_compiled = None

    def compile(self):
        self._obs_step_compiled = torch.compile(self._obs_step)
        self._img_step_compiled = torch.compile(self._img_step)
        return self

    def _obs_step(self, deter, stoch, prev_a, token, first):
        keep = (~first).to(token.dtype)[:, None]
        deter = self.core(deter * keep, stoch * keep[..., None], prev_a * keep)
        post = self.post_logits(deter, token)
        prior = self.prior_logits(deter)
        stoch, _ = self._dist_sample(post)
        return deter, stoch.to(token.dtype), post, prior

    def _img_step(self, deter, stoch, action):
        deter = self.core(deter, stoch, action)
        stoch, _ = self._dist_sample(self.prior_logits(deter))
        return deter, stoch.to(deter.dtype)

    def img_step(self, state, action):
        step = self._img_step_compiled or self._img_step
        deter, stoch = step(state["deter"], state["stoch"], action)
        return {"deter": deter, "stoch": stoch}

    def kl(self, post, prior):
        """KL(post || prior) per step, summed over the 32 categoricals; returns (dyn, rep) with stop-grads."""
        def probs(lg):
            p = torch.softmax(lg.float(), -1)
            return (1 - self.unimix) * p + self.unimix / self.classes

        p, q = probs(post), probs(prior)
        dyn = (p.detach() * (torch.log(p.detach()) - torch.log(q))).sum((-2, -1))
        rep = (p * (torch.log(p) - torch.log(q.detach()))).sum((-2, -1))
        return dyn, rep


class Encoder(nn.Module):
    def __init__(self, img_ch, vec_dim, depth, units, mults=(2, 3, 4, 4), layers=3, img=64):
        super().__init__()
        convs, c = [], img_ch
        for m in mults:
            convs.append(nn.ModuleDict({"conv": nn.Conv2d(c, depth * m, 5, padding=2), "norm": ChanRMSNorm(depth * m)}))
            c = depth * m
        self.convs = nn.ModuleList(convs)
        self.res = img // 2 ** len(mults)
        self.vec = mlp(vec_dim, units, layers)
        self.out_dim = c * self.res * self.res + units

    def forward(self, image, vector):  # image (N, H, W, C) uint8, vector (N, V)
        x = image.permute(0, 3, 1, 2).float() / 255.0 - 0.5
        for blk in self.convs:
            x = F.max_pool2d(blk["conv"](x), 2)
            x = F.silu(blk["norm"](x))
        return torch.cat([x.flatten(1), self.vec(symlog(vector))], -1)


class Decoder(nn.Module):
    def __init__(self, feat_dim, img_ch, vec_dim, depth, units, mults=(2, 3, 4, 4), layers=3, img=64):
        super().__init__()
        self.res = img // 2 ** len(mults)
        chans = [depth * m for m in mults][::-1]
        self.c0 = chans[0]
        self.lin = nn.Linear(feat_dim, self.c0 * self.res * self.res)
        self.norm0 = ChanRMSNorm(self.c0)
        ups, c = [], self.c0
        for o in chans[1:]:
            ups.append(nn.ModuleDict({"conv": nn.Conv2d(c, o, 5, padding=2), "norm": ChanRMSNorm(o)}))
            c = o
        self.ups = nn.ModuleList(ups)
        self.out = nn.Conv2d(c, img_ch, 5, padding=2)
        self.vec = mlp(feat_dim, units, layers)
        self.vec_out = nn.Linear(units, vec_dim)

    def forward(self, feat):  # (N, F) -> image (N, C, H, W) in [-0.5, 0.5] units, symlog vector
        x = self.lin(feat).reshape(-1, self.c0, self.res, self.res)
        x = F.silu(self.norm0(x))
        for blk in self.ups:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            x = F.silu(blk["norm"](blk["conv"](x)))
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.out(x), self.vec_out(self.vec(feat))


class Head(nn.Module):
    def __init__(self, i, units, layers, out, zero_init=True):
        super().__init__()
        self.net = mlp(i, units, layers)
        self.out = nn.Linear(units, out)
        if zero_init:
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    def forward(self, x):
        return self.out(self.net(x))


class Actor(nn.Module):
    def __init__(self, feat_dim, act_dim, units, layers=3, minstd=0.1, maxstd=1.0):
        super().__init__()
        self.net = mlp(feat_dim, units, layers)
        self.out = nn.Linear(units, 2 * act_dim)
        self.minstd, self.maxstd = minstd, maxstd

    def forward(self, feat):
        mean, std = self.out(self.net(feat)).float().chunk(2, -1)
        mean = torch.tanh(mean)
        std = (self.maxstd - self.minstd) * torch.sigmoid(std + 2.0) + self.minstd
        return torch.distributions.Normal(mean, std)
