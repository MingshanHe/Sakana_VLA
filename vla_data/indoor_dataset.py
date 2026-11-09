"""Minimal loader for the indoor subsets with OpenFly-v2-style action chunks.

Chunk at frame t (horizon H): the next H realized poses expressed in the yaw frame of frame t,
    chunk[k] = [R(-yaw_t) (p_{t+k+1} - p_t), wrap(yaw_{t+k+1} - yaw_t), stop_{t+k+1}],  k = 0..H-1
i.e. cumulative waypoints (not per-step deltas), padded past the episode end with the final pose and stop=1.
Executing a chunk = handing its waypoints to the same position controller the expert used; replanning
every `k_exec` steps re-anchors the chunk at the new frame.

    ds = IndoorVLADataset("data/vla_dataset/indoor_nano", horizon=10)
    item = ds[0]   # rgb (H,W,3) uint8, depth (H,W) float32 m, instruction str, state (4,), chunk (H,5), ...
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

CHUNK = 1000


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def action_chunks(states, stop, horizon):
    """states (T,4) = [x,y,z,yaw] (any common frame), stop (T,) -> (T, horizon, 5)."""
    T = len(states)
    idx = np.minimum(np.arange(T)[:, None] + np.arange(1, horizon + 1)[None], T - 1)
    fut = states[idx]                                   # (T,H,4)
    d = fut[..., :3] - states[:, None, :3]
    c, s = np.cos(states[:, 3])[:, None], np.sin(states[:, 3])[:, None]
    body = np.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1], d[..., 2]], -1)
    dyaw = wrap(fut[..., 3] - states[:, None, 3])
    st = stop[idx].astype(np.float32)
    st[np.arange(T)[:, None] + np.arange(1, horizon + 1)[None] >= T] = 1.0
    return np.concatenate([body, dyaw[..., None], st[..., None]], -1).astype(np.float32)


class IndoorVLADataset:
    def __init__(self, root, horizon=10, instruction_version="raw", with_depth=True):
        self.root = Path(root)
        self.horizon, self.with_depth = horizon, with_depth
        self.info = json.loads((self.root / "meta/info.json").read_text())
        self.episodes = [json.loads(l) for l in (self.root / "meta/episodes.jsonl").read_text().splitlines()]
        ipath = self.root / f"instructions/{instruction_version}.jsonl"
        self.instructions = {d["episode_index"]: d["instruction"]
                             for d in map(json.loads, ipath.read_text().splitlines())}
        self.index = [(e["episode_index"], t) for e in self.episodes for t in range(e["length"])]

    def __len__(self):
        return len(self.index)

    @lru_cache(maxsize=8)
    def _episode(self, ep):
        c = ep // CHUNK
        tab = pq.read_table(self.root / f"data/chunk-{c:03d}/episode_{ep:06d}.parquet").to_pydict()
        states = np.array(tab["observation.states"], np.float32)
        stop = np.array(tab["actions"], np.float32)[:, 4]
        out = {"states": states, "stop": stop, "boxes": np.array(tab["observation.boxes"], np.float32),
               "actions": np.array(tab["actions"], np.float32), "chunks": action_chunks(states, stop, self.horizon),
               "primitive": tab["primitive"], "rgb": self._video(ep)}
        if self.with_depth:
            out["depth"] = np.load(self.root / f"depth/chunk-{c:03d}/episode_{ep:06d}.npz")["depth"]
        return out

    def _video(self, ep):
        import imageio.v3 as iio
        key = "observation.images.rgb"
        p = self.root / f"videos/chunk-{ep // CHUNK:03d}/{key}/episode_{ep:06d}.mp4"
        return iio.imread(p, plugin="pyav") if _has_pyav() else _ffmpeg_decode(p)

    def __getitem__(self, i):
        ep, t = self.index[i]
        e = self._episode(ep)
        item = {"episode_index": ep, "frame_index": t, "instruction": self.instructions[ep],
                "rgb": e["rgb"][t], "state": e["states"][t], "box": e["boxes"][t], "action": e["actions"][t],
                "chunk": e["chunks"][t], "primitive": e["primitive"][t]}
        if self.with_depth:
            item["depth"] = e["depth"][t].astype(np.float32) / 1000.0
        return item


def _has_pyav():
    try:
        import av  # noqa: F401
        return True
    except ImportError:
        return False


def _ffmpeg_decode(path):
    import subprocess
    import imageio_ffmpeg
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    probe = subprocess.run([exe, "-i", str(path)], capture_output=True, text=True).stderr
    import re
    w, h = map(int, re.search(r", (\d{2,5})x(\d{2,5})", probe).groups())
    raw = subprocess.run([exe, "-loglevel", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)
