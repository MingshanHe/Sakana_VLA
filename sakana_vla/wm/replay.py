"""In-RAM replay of world-model episodes (npz files from export_replay.py / closed_loop --dump-wm)."""
import json
from pathlib import Path

import numpy as np
import torch

from .labels import scale_actions

KEYS = ("image", "vector", "action", "reward", "clearance", "target_vis", "at_goal", "is_first", "is_terminal")


def load_files(dirs):
    files = []
    for d in dirs:
        files += sorted(Path(d).glob("*.npz"))
    return files


def file_meta(f):
    with np.load(f) as d:
        return json.loads(str(d["meta"]))


class Replay:
    def __init__(self, files, seed=0):
        parts = {k: [] for k in KEYS}
        self.episodes = []
        for f in files:
            with np.load(f) as d:
                n = len(d["reward"])
                for k in KEYS:
                    parts[k].append(d[k])
                self.episodes.append({"file": str(f), "n": n, **json.loads(str(d["meta"]))})
        self.data = {k: np.concatenate(v, 0) for k, v in parts.items()}
        self.data["action"] = scale_actions(self.data["action"])
        self.n = len(self.data["reward"])
        self.rng = np.random.default_rng(seed)

    def sample(self, b, t):
        starts = self.rng.integers(0, self.n - t + 1, b)
        batch = {k: np.stack([self.data[k][s:s + t] for s in starts]) for k in KEYS}
        batch["is_first"] = batch["is_first"].copy()
        batch["is_first"][:, 0] = True
        return batch


def to_torch(batch, device):
    out = {}
    for k, v in batch.items():
        x = torch.from_numpy(np.ascontiguousarray(v)).to(device, non_blocking=True)
        out[k] = x if k == "image" else (x.bool() if k.startswith("is_") or k in ("target_vis", "at_goal") else x.float())
    return out
