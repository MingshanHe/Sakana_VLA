"""Export the expert dataset as world-model replay at the decision rate (env: sakana_hyperdrone, CPU).

    PYTHONPATH=. python -m sakana_vla.wm.export_replay --cache data/cache/v1 --out data/wm/expert --workers 6

One npz per episode (see labels.py for the step definition):
  image (K,64,64,4) uint8 RGB-D       vector (K,5) raw policy state        action (K,25) 5 increments, decision yaw frame
  reward (K,) geodesic progress achieved by action[k-1] (m, 0 at k=0)     clearance (K,) min clearance since step k-1 (m)
  target_vis (K,) bool   at_goal (K,) bool   stop (K,) expert stop flag    is_first (K,) bool   is_terminal (K,) bool
  geodesic (K,) m        meta: json (subset, episode_index, house, instruction)
"""
import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from datagen.geometry import build_maps, flight_mask
from datagen.planner import GoalField
from sakana_vla.data.dataset import FrameCache, chunk_increments, state_vector

from .labels import GOAL_DIST, VIS_PIXELS, WM_STEP, clearance_at, rgbd64

CACHE = None


def _init(cache_dir):
    global CACHE
    CACHE = FrameCache(cache_dir)


def subset_root(subset):
    for base in ("data/vla_dataset", "data/vla_parts"):
        if Path(base, subset).exists():
            return Path(base, subset)
    raise FileNotFoundError(subset)


def export_house(args):
    house, ep_ids, out_dir = args
    cache = CACHE
    scene = json.loads(Path(f"data/scenes/{house}.json").read_text())
    maps = build_maps(scene["glb"])
    mask = flight_mask(maps)
    extras = {}
    n_done, n_fallback = 0, 0
    for gi in ep_ids:
        e = cache.episodes[gi]
        subset, ei, off, n = e["subset"], e["episode_index"], e["offset"], e["length"]
        if subset not in extras:
            extras[subset] = {x["episode_index"]: x for x in map(json.loads, (subset_root(subset) / "meta/episodes_extra.jsonl").read_text().splitlines())}
        x = extras[subset][ei]
        goal = np.asarray(x["goal"])
        try:
            field = GoalField(maps, mask, maps["grid"].to_index(goal))
        except ValueError:
            field, n_fallback = None, n_fallback + 1
        root = subset_root(subset)
        depth = np.load(root / f"depth/chunk-{ei // 1000:03d}/episode_{ei:06d}.npz")["depth"]
        pix = np.array(pq.read_table(root / f"data/chunk-{ei // 1000:03d}/episode_{ei:06d}.parquet",
                                     columns=["observation.target_pixels"]).column(0).to_pylist())
        steps = np.arange(0, n, WM_STEP)
        K = len(steps)
        pos_all = cache.state_full[off:off + n, :3].astype(np.float64)  # world frame (observation.states is start-relative)
        geo = np.array([field.geodesic(pos_all[t]) if field is not None else np.linalg.norm(pos_all[t] - goal) for t in steps])
        clr_all = clearance_at(maps, pos_all)
        img = np.zeros((K, 64, 64, 4), np.uint8)
        vec = np.zeros((K, 5), np.float32)
        act = np.zeros((K, WM_STEP * 5), np.float32)
        clr = np.zeros(K, np.float32)
        for k, t in enumerate(steps):
            f = off + t
            img[k] = rgbd64(np.asarray(cache.frames[f]), depth[t])
            vec[k] = state_vector(cache, f)
            act[k] = chunk_increments(cache, f, WM_STEP).reshape(-1)
            clr[k] = clr_all[max(0, steps[k - 1] + 1) if k else t: t + 1].min() if k else clr_all[t]
        reward = np.zeros(K, np.float32)
        reward[1:] = geo[:-1] - geo[1:]
        vis = pix[steps] >= VIS_PIXELS
        at_goal = (np.linalg.norm(pos_all[steps] - goal, axis=1) <= GOAL_DIST) & vis
        is_first = np.zeros(K, bool)
        is_first[0] = True
        np.savez_compressed(out_dir / f"{subset}_{ei:06d}.npz", image=img, vector=vec, action=act, reward=reward, clearance=clr,
                            target_vis=vis, at_goal=at_goal, stop=cache.stop[off + steps].astype(bool), is_first=is_first,
                            is_terminal=np.zeros(K, bool), geodesic=geo.astype(np.float32),
                            meta=json.dumps({"subset": subset, "episode_index": ei, "house": house, "instruction": e["caption"],
                                             "source": "expert"}))
        n_done += 1
    return house, n_done, n_fallback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/cache/v1")
    ap.add_argument("--out", default="data/wm/expert")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--houses", nargs="*", default=None)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = FrameCache(args.cache)
    by_house = {}
    for gi, e in enumerate(cache.episodes):
        if args.houses is None or e["house"] in args.houses:
            by_house.setdefault(e["house"], []).append(gi)
    jobs = [(h, ids, out) for h, ids in sorted(by_house.items())]
    t0 = time.time()
    with Pool(args.workers, initializer=_init, initargs=(args.cache,)) as pool:
        for house, n_done, n_fb in pool.imap_unordered(export_house, jobs):
            print(f"{house}: {n_done} episodes ({n_fb} geodesic fallbacks)  [{time.time() - t0:.0f}s]", flush=True)
    files = sorted(out.glob("*.npz"))
    steps = sum(int(np.load(f)["reward"].shape[0]) for f in files[:50]) * len(files) // max(1, min(50, len(files)))
    print(f"done: {len(files)} episodes, ~{steps} steps, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
