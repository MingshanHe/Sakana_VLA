"""Generate a LeRobot v2.1 subset (e.g. indoor_nano / indoor_mini) from ProcTHOR houses.

    PYTHONPATH=. python -m datagen.generate --subset indoor_nano --houses 1-5 --episodes-per-house 40 \
        --out data/vla_dataset [--split train] [--seed 0]

Per house: maps -> sample (target, goal, start) -> batched dynamics rollout -> QA -> render -> labels -> write.
QA drops episodes that collide, lose tracking, never stop at the goal, or end without the target in view.
"""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from . import _jitfix  # noqa: F401  (must precede hyperlux use)
from . import label
from .expert_rollout import rollout
from .geometry import build_maps, flight_mask
from .render import HEIGHT, WIDTH, Renderer, depth_to_mm, target_bbox
from . import render as render_mod
from .simple_tasks import sample_simple_episodes
from .tasks import sample_episodes
from .write_dataset import DatasetWriter

MIN_END_PIXELS = 150      # target pixels in the final frame (256x256)
MIN_FRAMES, MAX_FRAMES = 20, 600


def parse_range(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def landmarks_of(seg, labels_map, target_type):
    vis = Counter()
    for s in seg[::5]:
        u, c = np.unique(s, return_counts=True)
        for k, v in zip(u, c):
            if v > 0.03 * s.size:
                vis[labels_map.get(str(int(k)), "")] += 1
    return [k for k, _ in vis.most_common() if k and k != target_type
            and not k.startswith(("wall", "floor", "ceiling", "door", "window", "room"))]


def keyframes_of(segs, n):
    ks = [("start", 0)] + [(s[0].replace(" ", "_"), s[1]) for s in segs[1:]] + [("end", n - 1)]
    seen, out = set(), []
    for name, t in ks:
        if t not in seen:
            seen.add(t)
            out.append((name, t))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--houses", required=True, help="ProcTHOR-Train indices, e.g. 1-5 or 1,3,7-9")
    ap.add_argument("--episodes-per-house", type=int, default=40)
    ap.add_argument("--per-target", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32, help="drones simulated in parallel")
    ap.add_argument("--perturb-prob", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/vla_dataset")
    ap.add_argument("--task-set", default="nav", choices=["nav", "simple"],
                    help="nav: cross-room navigation (v0). simple: same-room fly_to / land with a free straight line")
    ap.add_argument("--simple-tasks", default="fly_to,land")
    ap.add_argument("--cam-pitch", type=float, default=0.0, help="camera pitch down from the body x axis (deg)")
    args = ap.parse_args()
    render_mod.MOUNT_PITCH_DEG = args.cam_pitch

    from hyperdrone import dynamics
    rng = np.random.default_rng(args.seed)
    labels_map = json.loads(Path(__file__).with_name("class_labels.json").read_text())
    writer = DatasetWriter(args.out, args.subset, HEIGHT, WIDTH)
    sim = dynamics.Sim(args.batch, parameters={"dynamics": {"dynamics": {"model": "crazyflie"}}}, device="cpu",
                       control="position", seed=args.seed)
    report = {"args": vars(args), "houses": {}, "drop_reasons": Counter()}
    ep_index = 0
    t_all = time.time()
    for h in parse_range(args.houses):
        house = f"ProcTHOR-Train-{h}"
        t0 = time.time()
        meta = json.loads(Path(f"data/scenes/{house}.json").read_text())
        maps = build_maps(meta["glb"])
        mask = flight_mask(maps)
        room_types = {r["id"]: r["type"] for r in meta["rooms"]}
        specs = []
        sampler = (sample_simple_episodes(meta, maps, mask, rng, episodes_per_target=args.per_target,
                                          tasks=tuple(args.simple_tasks.split(",")))
                   if args.task_set == "simple" else
                   sample_episodes(meta, maps, mask, rng, episodes_per_target=args.per_target))
        if args.task_set == "simple":  # balance: up to half land episodes (few objects are landable), rest fly_to
            pool = []
            for sf in sampler:
                pool.append(sf)
                if len(pool) >= 4 * args.episodes_per_house:
                    break
            land = [sf for sf in pool if sf[0]["task"] == "land"][:args.episodes_per_house // 2]
            other = [sf for sf in pool if sf[0]["task"] != "land"][:args.episodes_per_house - len(land)]
            specs = land + other
            specs = [specs[i] for i in rng.permutation(len(specs))]
            sampler = ()
        for sf in sampler:
            specs.append(sf)
            if len(specs) >= args.episodes_per_house:
                break
        eps = []
        for b in range(0, len(specs), args.batch):
            eps += rollout(specs[b:b + args.batch], sim, maps, rng, perturb_prob=args.perturb_prob)
        renderer = Renderer(meta["glb"], batch=64)
        kept = 0
        drops = Counter()
        for e in eps:
            if e.failed:
                drops[e.failed] += 1
                continue
            arr = label.frame_arrays(e.records)
            n = len(e.records)
            task = e.spec.get("task", "nav")
            stop = label.stop_flags(arr, np.asarray(e.spec.get("stop_point", e.spec["goal"])),
                                    np.asarray(e.spec["target"]["center"]), face_target=task != "land")
            if not (MIN_FRAMES <= n <= MAX_FRAMES):
                drops["length"] += 1
                continue
            if stop.sum() < 3:
                drops["no_stop"] += 1
                continue
            rgb, depth, seg = renderer.render(arr["position"], arr["orientation"])
            boxes, pix = target_bbox(seg, e.spec["target"]["segmentation_class"])
            if task != "land" and pix[-1] < MIN_END_PIXELS:  # when landed, the target is below the camera
                drops["target_not_visible"] += 1
                continue
            dvalid = depth[np.isfinite(depth) & (depth > 0)]
            if dvalid.size == 0 or dvalid.min() < 0.05 or rgb.reshape(n, -1).max(1).min() < 10:
                drops["bad_frames"] += 1
                continue
            acts = label.actions_from_states(arr["position"], arr["yaw"], stop)
            prims = label.primitives(arr, stop)
            segs = label.segments(prims)
            rooms_seq = label.room_sequence(meta, arr["position"])
            landmarks = landmarks_of(seg, labels_map, e.spec["target"]["type"])
            instruction = e.spec.get("instruction") or label.instruction_draft(e.spec, room_types, rooms_seq, landmarks)
            extra = {
                "house": house, "target": e.spec["target"], "goal": e.spec["goal"], "start": e.spec["start"],
                "geodesic": e.spec["geodesic"], "rooms": [{"id": r, "type": room_types.get(r)} for r in rooms_seq],
                "segments": [{"primitive": s[0], "start": s[1], "end": s[2]} for s in segs],
                "landmarks": landmarks[:8], "perturbed": bool(e.perturbed),
                "track_err_mean": float(np.mean(e.track_err)), "track_err_max": float(np.max(e.track_err)),
                "path_length": float(np.linalg.norm(np.diff(arr["position"], axis=0), axis=1).sum()),
                "final_target_pixels": int(pix[-1]), "split": args.subset,
                "task": task, "cam_pitch_deg": args.cam_pitch, "start_target_pixels": int(pix[0]),
                "target_visible_frac": float((pix >= 20).mean()),
                **{k: e.spec[k] for k in ("start_mode", "stop_point", "distance") if k in e.spec},
            }
            writer.add(ep_index, arr, acts, stop, prims, boxes, pix, rgb, depth_to_mm(depth), e.spec, extra,
                       instruction, keyframes_of(segs, n))
            ep_index += 1
            kept += 1
        del renderer
        report["houses"][house] = {"sampled": len(specs), "kept": kept, "drops": dict(drops),
                                   "seconds": round(time.time() - t0, 1)}
        report["drop_reasons"].update(drops)
        print(f"{house}: kept {kept}/{len(specs)} drops {dict(drops)} ({time.time() - t0:.0f}s)", flush=True)
    writer.close()
    report.update({"episodes": ep_index, "frames": writer.total_frames, "seconds": round(time.time() - t_all, 1)})
    (writer.dir / "meta" / "generation_report.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({k: report[k] for k in ("episodes", "frames", "seconds", "drop_reasons")}, default=str))


if __name__ == "__main__":
    main()
