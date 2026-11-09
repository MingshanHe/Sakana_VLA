"""Server-side chunk verifier: keeps one RSSM posterior per episode and scores candidate chunks in imagination.

Used by eval/policy_server.py (env sakana_vla2, GPU). Per decision: observe(current RGB-D, state, the 5 increments
executed since the last decision) -> posterior; score(candidates) -> imagine the first S x 5 increments of each
candidate and combine the predicted outcomes (progress, collision, clearance, target visible). A stop candidate is
scored from the predicted "at goal" probability of the current state.
"""
import numpy as np
import torch

from .dv3.nets import TwoHot
from .labels import chunk_to_wm_actions, scale_actions
from .model import ChunkWorldModel, combine_scores


class ChunkVerifier:
    def __init__(self, ckpt, device="cuda", lam_col=10.0, lam_clr=5.0, mu_vis=0.5, stop_bonus=2.0, steps=2):
        ck = torch.load(ckpt, map_location=device, weights_only=False)
        self.wm = ChunkWorldModel(**ck["cfg"]).to(device).eval()
        self.wm.load_state_dict(ck["wm"])
        self.twohot = TwoHot(device=device)
        self.device, self.states = device, {}
        self.lam_col, self.lam_clr, self.mu_vis, self.stop_bonus, self.steps = lam_col, lam_clr, mu_vis, stop_bonus, steps
        self.step = ck.get("step")

    def _gather(self, ids):
        init = self.wm.rssm.initial(1, self.device)
        deter = torch.stack([self.states[i]["deter"] if i in self.states else init["deter"][0] for i in ids])
        stoch = torch.stack([self.states[i]["stoch"] if i in self.states else init["stoch"][0] for i in ids])
        return {"deter": deter, "stoch": stoch}

    @torch.no_grad()
    def observe(self, ids, obs, vectors, prev_actions, is_first):
        img = torch.as_tensor(np.asarray(obs), device=self.device)
        vec = torch.as_tensor(np.asarray(vectors, np.float32), device=self.device)
        a = torch.as_tensor(scale_actions(prev_actions), device=self.device)
        first = torch.as_tensor(np.asarray(is_first, bool), device=self.device)
        st = self._gather(ids)
        with torch.autocast("cuda", torch.bfloat16):
            tokens = self.wm.enc(img, vec)
            deter, stoch, _, _ = self.wm.rssm._obs_step(st["deter"], st["stoch"], a, tokens, first)
        for k, i in enumerate(ids):
            self.states[i] = {"deter": deter[k].float(), "stoch": stoch[k].float()}

    @torch.no_grad()
    def at_goal(self, ids):
        st = self._gather(ids)
        with torch.autocast("cuda", torch.bfloat16):
            return self.wm.outcomes(self.wm.feat(st), self.twohot)["at_goal"].float().cpu().numpy()

    @torch.no_grad()
    def score(self, ids, cands):
        """cands (B, N, H, 5) raw increments -> scores (B, N), at-goal probability (B,), parts dict of (B, N) arrays."""
        B, N = cands.shape[:2]
        acts = np.stack([[chunk_to_wm_actions(cands[b, k], self.steps) for k in range(N)] for b in range(B)])
        acts = torch.as_tensor(scale_actions(acts), device=self.device).reshape(B * N, self.steps, -1)
        st = self._gather(ids)
        st = {k: v.repeat_interleave(N, 0) for k, v in st.items()}
        with torch.autocast("cuda", torch.bfloat16):
            o = self.wm.rollout_outcomes(st, acts, self.twohot)
            score = combine_scores(o, self.lam_col, self.lam_clr, self.mu_vis).reshape(B, N).float()
            goal_now = self.wm.outcomes(self.wm.feat(self._gather(ids)), self.twohot)["at_goal"].float()
        score = score.cpu().numpy()
        goal_now = goal_now.cpu().numpy()
        stop = cands[:, :, 0, 4] > 0.5
        score[stop] = (self.stop_bonus * (2 * goal_now[:, None] - 1) * np.ones((B, N)))[stop]
        parts = {k: v.reshape(B, N).float().cpu().numpy() for k, v in o.items()}
        return score, goal_now, parts
