"""Decode LeRobot subsets into one uint8 memmap + per-frame arrays for fast random-access training.

    python -m sakana_vla.data.frames_cache --subsets data/vla_dataset/indoor_nano data/vla_parts/indoor_mini_train_a \
        --out data/cache/train_v0
Writes <out>/frames.u8 (N, H, W, 3) and <out>/meta.npz (+ meta.json with captions per episode).
"""
import argparse
import json
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq


def decode(path):
    with av.open(str(path)) as c:
        return np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)])


def build(subsets, out, h=256, w=256):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    eps = []
    for root in subsets:
        root = Path(root)
        for line in (root / "meta/episodes.jsonl").read_text().splitlines():
            e = json.loads(line)
            eps.append((root, e["episode_index"], e["length"]))
    n_total = sum(l for _, _, l in eps)
    frames = np.lib.format.open_memmap(out / "frames.npy", mode="w+", dtype=np.uint8, shape=(n_total, h, w, 3))
    cols = {"states": [], "state_full": [], "stop": [], "ep": [], "t": []}
    episodes = []
    off = 0
    for gi, (root, ei, n) in enumerate(eps):
        c = ei // 1000
        tab = pq.read_table(root / f"data/chunk-{c:03d}/episode_{ei:06d}.parquet").to_pydict()
        rgb = decode(root / f"videos/chunk-{c:03d}/observation.images.rgb/episode_{ei:06d}.mp4")
        if len(rgb) != n:
            raise RuntimeError(f"{root} ep {ei}: {len(rgb)} frames decoded, {n} expected")
        frames[off:off + n] = rgb
        cols["states"].append(np.array(tab["observation.states"], np.float32))
        cols["state_full"].append(np.array(tab["observation.state_full"], np.float32))
        cols["stop"].append(np.array(tab["actions"], np.float32)[:, 4])
        cols["ep"].append(np.full(n, gi, np.int32))
        cols["t"].append(np.arange(n, dtype=np.int32))
        episodes.append({"subset": root.name, "episode_index": ei, "offset": off, "length": n,
                         "house": tab["env"][0], "caption": tab["caption"][0]})
        off += n
        if gi % 100 == 0:
            print(f"{gi}/{len(eps)} episodes, {off}/{n_total} frames", flush=True)
    frames.flush()
    np.savez(out / "meta.npz", **{k: np.concatenate(v) for k, v in cols.items()})
    (out / "episodes.json").write_text(json.dumps(episodes))
    print(f"wrote {out}: {len(eps)} episodes, {n_total} frames")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    build(a.subsets, a.out)
