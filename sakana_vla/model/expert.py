# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
# Copyright 2026 Junli Liu (AeroBrain-project / OpenFlyV2 action Expert design).
# Modified 2026 for Sakana_VLA: no MAM / MSP / router head, read-only prefix KV.
# SPDX-License-Identifier: Apache-2.0
"""Action Expert: a narrow Qwen3.5 decoder that mirrors the VLM's hybrid layer schedule.

The Expert's full-attention layer i attends to [VLM layer-i prefix K/V ; suffix K/V] (pi0-style shared attention).
Its query heads equal the VLM's KV heads (2 x 256), so prefix K/V plug in without projection. Linear-attention
(Gated DeltaNet) layers run on the suffix alone (OpenFlyV2 default: share_linear_attention_state=False).
Suffix = [state token, H action tokens]; all suffix tokens see each other and every valid prefix token.
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention,
    Qwen3_5GatedDeltaNet,
    Qwen3_5MLP,
    Qwen3_5RMSNorm,
    Qwen3_5TextRotaryEmbedding,
)


class PrefixKV:
    """Read-only stand-in for a transformers Cache: update() returns [prefix ; new] without storing anything,
    so the same prefix serves every flow-integration step."""

    def __init__(self, keys, values):
        self.keys, self.values = keys, values  # dict layer_idx -> (B, kv_heads, Lp, head_dim)

    @classmethod
    def from_vlm_cache(cls, cache, layer_types):
        keys, values = {}, {}
        for i, t in enumerate(layer_types):
            if t == "full_attention":
                keys[i], values[i] = cache.layers[i].keys, cache.layers[i].values
        return cls(keys, values)

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        k, v = self.keys[layer_idx], self.values[layer_idx]
        return (torch.cat([k.to(key_states.dtype), key_states], dim=-2),
                torch.cat([v.to(value_states.dtype), value_states], dim=-2))

    def detach(self):
        return PrefixKV({i: k.detach() for i, k in self.keys.items()}, {i: v.detach() for i, v in self.values.items()})

    def repeat_interleave(self, n):
        return PrefixKV({i: k.repeat_interleave(n, 0) for i, k in self.keys.items()},
                        {i: v.repeat_interleave(n, 0) for i, v in self.values.items()})

    @property
    def length(self):
        return next(iter(self.keys.values())).shape[-2]


def make_expert_config(vlm_text_config, hidden_size=640):
    """OpenFlyV2 recipe: copy the VLM text config, shrink width; attention heads = VLM KV heads (same head_dim)."""
    cfg = copy.deepcopy(vlm_text_config)
    vlm_h, vlm_inter = cfg.hidden_size, cfg.intermediate_size
    cfg.hidden_size = hidden_size
    cfg.num_attention_heads = cfg.num_key_value_heads
    cfg.intermediate_size = round(vlm_inter * hidden_size / vlm_h / 64) * 64
    cfg._attn_implementation = "sdpa"
    return cfg


class ExpertLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_idx)
        else:
            self.self_attn = Qwen3_5Attention(config, layer_idx)
        self.mlp = Qwen3_5MLP(config, config.intermediate_size)
        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, h, position_embeddings, attn_mask, prefix_kv):
        r = h
        h = self.input_layernorm(h)
        if self.layer_type == "linear_attention":
            h = self.linear_attn(hidden_states=h, cache_params=None, attention_mask=None)
        else:
            h, _ = self.self_attn(hidden_states=h, position_embeddings=position_embeddings,
                                  attention_mask=attn_mask, past_key_values=prefix_kv)
        h = r + h
        return h + self.mlp(self.post_attention_layernorm(h))


def sinusoidal_time_embedding(t, dim, min_period=4e-3, max_period=4.0):
    """pi0-style sinusoidal embedding of flow time t in [0, 1]."""
    frac = torch.linspace(0.0, 1.0, dim // 2, device=t.device, dtype=torch.float64)
    period = min_period * (max_period / min_period) ** frac
    x = (2 * torch.pi / period)[None] * t.double()[:, None]
    return torch.cat([torch.sin(x), torch.cos(x)], dim=1).float()


class ActionExpert(nn.Module):
    def __init__(self, config, action_dim, state_dim):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([ExpertLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(config=config)
        d = config.hidden_size
        self.action_in_proj = nn.Linear(action_dim, d)
        self.state_proj = nn.Linear(state_dim, d)
        self.action_time_mlp_in = nn.Linear(2 * d, d)
        self.action_time_mlp_out = nn.Linear(d, d)
        self.action_out_proj = nn.Linear(d, action_dim)
        self.gradient_checkpointing = False
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=self.config.initializer_range)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.action_out_proj.weight)  # start by predicting zero velocity

    def embed_suffix(self, noisy_actions, states, t):
        d = self.config.hidden_size
        a = self.action_in_proj(noisy_actions)                                  # (B, H, d)
        te = sinusoidal_time_embedding(t, d).to(a.dtype)[:, None].expand_as(a)
        at = self.action_time_mlp_out(F.silu(self.action_time_mlp_in(torch.cat([a, te], -1))))
        s = self.state_proj(states)[:, None]                                    # (B, 1, d)
        return torch.cat([s, at], dim=1)

    def forward(self, suffix, prefix_kv, prefix_mask, position_ids):
        """suffix (B, Ls, d); prefix_mask (B, Lp) 1=valid; position_ids (B, Ls) text positions after the prefix."""
        B, Ls, _ = suffix.shape
        # boolean SDPA mask (True = attend): valid prefix tokens + the whole suffix (bidirectional)
        sees_prefix = prefix_mask[:, None, None, :].bool().expand(B, 1, Ls, prefix_mask.shape[1])
        attn_mask = torch.cat([sees_prefix, torch.ones(B, 1, Ls, Ls, device=suffix.device, dtype=torch.bool)], -1)
        pos = self.rotary_emb(suffix, position_ids)
        h = suffix
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                h = torch.utils.checkpoint.checkpoint(layer, h, pos, attn_mask, prefix_kv, use_reentrant=False)
            else:
                h = layer(h, pos, attn_mask, prefix_kv)
        return self.norm(h)
