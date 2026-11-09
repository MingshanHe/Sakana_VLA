"""Training samples: (current RGB, instruction, state, normalized H-step action chunk).

Chunk target at frame t: per-step increments in the yaw frame of frame t,
    inc[k] = [R(-yaw_t)(p_{t+k+1} - p_{t+k}), yaw_{t+k+1} - yaw_{t+k}, stop_{t+k+1}],  k = 0..H-1,
padded past the episode end with zero motion and stop=1. Execution: cumsum -> waypoints relative to frame t.
State (5): body-frame velocity (vx, vy, vz), yaw rate, height.
Normalization: xyz/yaw increments scaled by robust per-dim bounds (q0.5/q99.5) to [-1, 1] (clipped); stop -> 2s-1.
"""
import json
from pathlib import Path

import numpy as np
import torch

PROMPT = "You are an indoor drone. Instruction: {instruction}\nPredict the next flight actions."


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def yaw_from_quat(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class FrameCache:
    def __init__(self, root):
        self.root = Path(root)
        self.frames = np.load(self.root / "frames.npy", mmap_mode="r")
        m = np.load(self.root / "meta.npz")
        self.states, self.state_full, self.stop = m["states"], m["state_full"], m["stop"]
        self.ep, self.t = m["ep"], m["t"]
        self.episodes = json.loads((self.root / "episodes.json").read_text())

    def __len__(self):
        return len(self.ep)


def chunk_increments(cache, i, horizon):
    """(horizon, 5) raw increments for global frame index i."""
    e = cache.episodes[cache.ep[i]]
    off, n, t = e["offset"], e["length"], cache.t[i]
    idx = off + np.minimum(t + np.arange(horizon + 1), n - 1)       # frames t..t+H (clamped)
    s = cache.states[idx].astype(np.float64)                          # [x, y, z, yaw] (start frame)
    yaw0 = s[0, 3]
    d = np.diff(s[:, :3], axis=0)
    c, si = np.cos(yaw0), np.sin(yaw0)
    body = np.stack([c * d[:, 0] + si * d[:, 1], -si * d[:, 0] + c * d[:, 1], d[:, 2]], -1)
    dyaw = wrap(np.diff(s[:, 3]))
    stop = cache.stop[idx[1:]].astype(np.float64)
    past_end = (t + np.arange(1, horizon + 1)) >= n
    stop[past_end] = 1.0
    out = np.c_[body, dyaw, stop]
    out[stop > 0.5, :4] = 0.0
    return out.astype(np.float32)


def state_vector(cache, i):
    f = cache.state_full[i].astype(np.float64)
    yaw = yaw_from_quat(f[3:7])
    c, s = np.cos(yaw), np.sin(yaw)
    vx, vy, vz = f[7:10]
    return np.array([c * vx + s * vy, -s * vx + c * vy, vz, f[12], f[2]], np.float32)


def compute_norm_stats(cache, indices, horizon, n_sample=20000, seed=0):
    rng = np.random.default_rng(seed)
    pick = rng.choice(indices, size=min(n_sample, len(indices)), replace=False)
    inc = np.concatenate([chunk_increments(cache, i, horizon) for i in pick])
    st = np.stack([state_vector(cache, i) for i in pick])
    moving = inc[inc[:, 4] < 0.5, :4]
    lo, hi = np.percentile(moving, 0.5, axis=0), np.percentile(moving, 99.5, axis=0)
    bound = np.maximum(np.abs(lo), np.abs(hi))                         # symmetric: 0 motion -> 0
    return {"action_bound": bound.tolist(), "state_mean": st.mean(0).tolist(), "state_std": (st.std(0) + 1e-6).tolist(),
            "horizon": horizon}


def normalize_actions(inc, stats):
    out = inc.copy()
    out[..., :4] = np.clip(inc[..., :4] / np.asarray(stats["action_bound"], np.float32), -1, 1)
    out[..., 4] = inc[..., 4] * 2 - 1
    return out


def denormalize_actions(x, stats):
    x = np.asarray(x, np.float32).copy()
    x[..., :4] = np.clip(x[..., :4], -1, 1) * np.asarray(stats["action_bound"], np.float32)
    x[..., 4] = (np.clip(x[..., 4], -1, 1) + 1) / 2
    return x


class VLADataset(torch.utils.data.Dataset):
    def __init__(self, cache, indices, stats, horizon=12, blind=False):
        self.cache, self.indices, self.stats, self.horizon, self.blind = cache, np.asarray(indices), stats, horizon, blind
        self.s_mean = np.asarray(stats["state_mean"], np.float32)
        self.s_std = np.asarray(stats["state_std"], np.float32)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        i = int(self.indices[k])
        img = np.array(self.cache.frames[i])
        if self.blind:
            img[:] = 0
        inc = chunk_increments(self.cache, i, self.horizon)
        return {"image": img, "instruction": self.cache.episodes[self.cache.ep[i]]["caption"],
                "state": (state_vector(self.cache, i) - self.s_mean) / self.s_std,
                "actions": normalize_actions(inc, self.stats), "raw_actions": inc, "index": i}


class Collator:
    def __init__(self, processor):
        self.p = processor
        self.p.tokenizer.padding_side = "right"

    def __call__(self, items):
        texts = []
        for it in items:
            msgs = [{"role": "user", "content": [{"type": "image"},
                                                 {"type": "text", "text": PROMPT.format(instruction=it["instruction"])}]}]
            texts.append(self.p.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        enc = self.p(text=texts, images=[it["image"] for it in items], return_tensors="pt", padding=True)
        out = {k: v for k, v in enc.items()}
        out["states"] = torch.from_numpy(np.stack([it["state"] for it in items]))
        out["actions"] = torch.from_numpy(np.stack([it["actions"] for it in items]))
        out["raw_actions"] = torch.from_numpy(np.stack([it["raw_actions"] for it in items]))
        out["index"] = torch.tensor([it["index"] for it in items])
        return out


def split_indices(cache, val_frac=0.05, unseen_subsets=("indoor_mini_unseen",), unseen_houses=(), seed=0):
    """Episode-level split: train / val_seen (held-out episodes of training houses) / val_unseen (unseen houses)."""
    rng = np.random.default_rng(seed)
    unseen_eps = {i for i, e in enumerate(cache.episodes) if e["subset"] in unseen_subsets or e["house"] in unseen_houses}
    seen_eps = [i for i in range(len(cache.episodes)) if i not in unseen_eps]
    val_eps = set(rng.choice(seen_eps, size=max(1, int(len(seen_eps) * val_frac)), replace=False).tolist())
    ep = cache.ep
    train = np.flatnonzero(~np.isin(ep, list(unseen_eps | val_eps)))
    val_seen = np.flatnonzero(np.isin(ep, list(val_eps)))
    val_unseen = np.flatnonzero(np.isin(ep, list(unseen_eps)))
    return train, val_seen, val_unseen
