"""SakanaVLA v0: Qwen3.5-0.8B prefix + flow-matching action Expert (OpenFlyV2 recipe without MAM / MSP).

Prefix: [image tokens (current RGB), instruction tokens] through the Qwen3.5 VLM (frozen in stage A).
Suffix: [state token, H noisy action tokens (+ flow time)] through the Expert, attending to the prefix K/V of
every full-attention layer. Flow matching: x_t = t*z + (1-t)*a, target u = z - a, t ~ 0.999*Beta(1.5,1)+0.001,
z ~ N(0, I) (no motion prior). Inference integrates from t=1 to 0 with Euler steps.
"""
from contextlib import nullcontext

import torch
import torch.nn as nn
from transformers import Qwen3_5ForConditionalGeneration

from .expert import ActionExpert, PrefixKV, make_expert_config


class SakanaVLA(nn.Module):
    def __init__(self, vlm_path, action_dim=5, state_dim=5, horizon=12, expert_hidden=640, freeze_vlm=True,
                 stop_weight=1.0, vlm_dtype=torch.bfloat16):
        super().__init__()
        self.vlm = Qwen3_5ForConditionalGeneration.from_pretrained(vlm_path, dtype=vlm_dtype)
        self.vlm.lm_head = nn.Identity()  # unused (tied to embeddings anyway)
        self.text_cfg = self.vlm.config.text_config
        self.expert = ActionExpert(make_expert_config(self.text_cfg, expert_hidden), action_dim, state_dim)
        self.horizon, self.action_dim = horizon, action_dim
        self.freeze_vlm = freeze_vlm
        self.register_buffer("channel_weight", torch.ones(action_dim), persistent=False)
        self.channel_weight[-1] = stop_weight
        if freeze_vlm:
            self.vlm.requires_grad_(False)
            self.vlm.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_vlm:
            self.vlm.eval()
        return self

    def encode_prefix(self, input_ids, attention_mask, pixel_values, image_grid_thw, mm_token_type_ids=None):
        ctx = torch.no_grad() if self.freeze_vlm else nullcontext()
        kwargs = {"mm_token_type_ids": mm_token_type_ids} if mm_token_type_ids is not None else {}
        with ctx:
            out = self.vlm.model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values,
                                 image_grid_thw=image_grid_thw, use_cache=True, **kwargs)
        kv = PrefixKV.from_vlm_cache(out.past_key_values, self.text_cfg.layer_types)
        if self.freeze_vlm:
            kv = kv.detach()
        # text position of the last valid prefix token (right padding): (valid_len - 1) + mrope delta
        last = attention_mask.sum(1) - 1 + out.rope_deltas.reshape(-1).to(attention_mask.device)
        return kv, last.long()

    def _expert(self, x_t, states, t, kv, prefix_mask, last_pos):
        suffix = self.expert.embed_suffix(x_t, states, t)
        pos = last_pos[:, None] + torch.arange(1, suffix.shape[1] + 1, device=suffix.device)[None]
        h = self.expert(suffix, kv, prefix_mask, pos)
        return self.expert.action_out_proj(h[:, -self.horizon:])

    @staticmethod
    def sample_time(b, device):
        t = torch.distributions.Beta(torch.tensor(1.5, device=device), torch.tensor(1.0, device=device)).sample((b,))
        return t * 0.999 + 0.001

    def forward(self, batch):
        kv, last = self.encode_prefix(batch["input_ids"], batch["attention_mask"], batch["pixel_values"],
                                      batch["image_grid_thw"], batch.get("mm_token_type_ids"))
        a, s = batch["actions"], batch["states"]
        t = self.sample_time(a.shape[0], a.device)
        z = torch.randn_like(a)
        tt = t[:, None, None]
        x_t = tt * z + (1 - tt) * a
        v = self._expert(x_t, s, t, kv, batch["attention_mask"], last)
        err = (v.float() - (z - a).float()).square() * self.channel_weight
        loss = err.sum(-1).mean() / self.channel_weight.sum()
        with torch.no_grad():
            metrics = {"fm_xyz": err[..., :3].mean().item(), "fm_yaw": err[..., 3].mean().item(),
                       "fm_stop": err[..., 4].mean().item() if a.shape[-1] > 4 else 0.0}
        return loss, metrics

    @torch.no_grad()
    def sample(self, batch, steps=5, noise=None):
        kv, last = self.encode_prefix(batch["input_ids"], batch["attention_mask"], batch["pixel_values"],
                                      batch["image_grid_thw"], batch.get("mm_token_type_ids"))
        B = batch["input_ids"].shape[0]
        x = noise if noise is not None else torch.randn(B, self.horizon, self.action_dim, device=last.device)
        dt = -1.0 / steps
        for k in range(steps):
            t = torch.full((B,), 1.0 + k * dt, device=x.device)
            x = x + dt * self._expert(x, batch["states"], t, kv, batch["attention_mask"], last).float()
        return x
