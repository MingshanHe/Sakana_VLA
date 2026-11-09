"""S0 spike: one house end to end (plan -> dynamics expert -> labels -> render) with diagnostics.

    PYTHONPATH=. python -m datagen.spike --house ProcTHOR-Train-3 --episodes 8 --out data/spike
Writes top-down plots, per-episode mp4 (RGB | depth | bbox overlay) and a stats json.
"""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from . import _jitfix  # noqa: F401  (must precede hyperlux use)
from . import label
from .expert_rollout import rollout
from .geometry import build_maps, flight_mask
from .tasks import sample_episodes


def topdown(maps, mask, eps, path):
    g = maps["grid"]
    img = np.where(mask.any(2), 235, 60).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)

    def px(p):
        ij = (np.asarray(p)[..., :2] - g.origin[:2]) / g.res * 4
        return np.stack([ij[..., 1], ij[..., 0]], -1).astype(np.int32)  # (row=i -> y image), draw as (x=j, y=i)

    colors = [(255, 0, 0), (0, 160, 0), (0, 0, 255), (200, 0, 200), (0, 160, 200), (200, 120, 0), (90, 90, 90), (0, 90, 255)]
    for n, e in enumerate(eps):
        c = colors[n % len(colors)]
        if e.records:
            real = np.array([r["position"] for r in e.records])
            cv2.polylines(img, [px(real)], False, c, 1)
            pert = [r["position"] for r in e.records if r["perturbed"]]
            for p in pert:
                cv2.circle(img, tuple(px(p)), 2, (0, 0, 0), -1)
        cv2.circle(img, tuple(px(e.spec["start"]["position"])), 4, c, -1)
        cv2.drawMarker(img, tuple(px(e.spec["goal"])), c, cv2.MARKER_STAR, 10, 2)
        lo, hi = np.array(e.spec["target"]["aabb"])
        cv2.rectangle(img, tuple(px(lo)), tuple(px(hi)), c, 1)
    cv2.imwrite(str(path), img)


def overlay_video(rgb, depth, boxes, prims, stop, text, path, fps=10):
    frames = []
    h, w = rgb.shape[1:3]
    for t in range(len(rgb)):
        a = cv2.cvtColor(rgb[t], cv2.COLOR_RGB2BGR).copy()
        x, y, bw, bh = boxes[t]
        if bw > 0:
            cv2.rectangle(a, (int(x * w), int(y * h)), (int((x + bw) * w), int((y + bh) * h)), (0, 255, 0), 1)
        cv2.putText(a, f"{t:03d} {prims[t]}{' STOP' if stop[t] else ''}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        d = cv2.applyColorMap((np.clip(np.nan_to_num(depth[t]) / 5.0, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        frame = np.concatenate([a, d], 1)
        bar = np.zeros((28, frame.shape[1], 3), np.uint8)
        cv2.putText(bar, text[:95], (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        frames.append(cv2.cvtColor(np.concatenate([frame, bar], 0), cv2.COLOR_BGR2RGB))
    imageio.mimsave(path, frames, fps=fps, macro_block_size=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--house", default="ProcTHOR-Train-3")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--per-target", type=int, default=2)
    ap.add_argument("--perturb-prob", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/spike")
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    timing = {}

    meta = json.loads(Path(f"data/scenes/{args.house}.json").read_text())
    t = time.time()
    maps = build_maps(meta["glb"])
    mask = flight_mask(maps)
    timing["maps"] = time.time() - t

    t = time.time()
    specs = []
    for spec, field in sample_episodes(meta, maps, mask, rng, episodes_per_target=args.per_target):
        specs.append((spec, field))
        if len(specs) >= args.episodes:
            break
    timing["sample+dijkstra"] = time.time() - t

    from hyperdrone import dynamics
    t = time.time()
    sim = dynamics.Sim(len(specs), parameters={"dynamics": {"dynamics": {"model": "crazyflie"}}}, device="cpu",
                       control="position", seed=args.seed)
    eps = rollout(specs, sim, maps, rng, perturb_prob=args.perturb_prob)
    timing["rollout"] = time.time() - t
    topdown(maps, mask, eps, out / "topdown.png")

    renderer = None
    if not args.no_render:
        from .render import Renderer, target_bbox
        t = time.time()
        renderer = Renderer(meta["glb"], batch=32)
        timing["renderer_init"] = time.time() - t

    room_types = {r["id"]: r["type"] for r in meta["rooms"]}
    labels_map = json.loads((Path(__file__).with_name("class_labels.json")).read_text())
    stats = []
    t_render = 0.0
    for n, e in enumerate(eps):
        row = {"episode": n, "target": e.spec["target"]["type"], "geodesic": round(e.spec["geodesic"], 2),
               "failed": e.failed, "frames": len(e.records), "perturbed": bool(e.perturbed),
               "track_err_mean": round(float(np.mean(e.track_err)), 3) if e.track_err else None,
               "track_err_max": round(float(np.max(e.track_err)), 3) if e.track_err else None}
        if e.failed or not e.records:
            stats.append(row)
            continue
        arr = label.frame_arrays(e.records)
        stop = label.stop_flags(arr, np.asarray(e.spec["goal"]), np.asarray(e.spec["target"]["center"]))
        acts = label.actions_from_states(arr["position"], arr["yaw"], stop)
        prims = label.primitives(arr, stop)
        segs = label.segments(prims)
        rooms_seq = label.room_sequence(meta, arr["position"])
        row.update({"stop_frames": int(stop.sum()), "segments": [(s[0], s[2] - s[1]) for s in segs],
                    "rooms": [room_types.get(r) for r in rooms_seq],
                    "final_dist": round(float(np.linalg.norm(arr["position"][-1] - e.spec["goal"])), 3),
                    "action_abs_max": np.abs(acts[:, :4]).max(0).round(3).tolist(),
                    "speed_max": round(float(np.linalg.norm(arr["linear_velocity"], axis=1).max()), 2),
                    "tilt_max_deg": round(float(np.degrees(np.arccos(np.clip(1 - 2 * (arr["orientation"][:, 1] ** 2 + arr["orientation"][:, 2] ** 2), -1, 1))).max()), 1)})
        landmarks = []
        if renderer is not None:
            t = time.time()
            rgb, depth, seg = renderer.render(arr["position"], arr["orientation"])
            t_render += time.time() - t
            boxes, pix = target_bbox(seg, e.spec["target"]["segmentation_class"])
            vis = Counter()
            for s in seg[::5]:
                u, c = np.unique(s, return_counts=True)
                for k, v in zip(u, c):
                    if v > 0.03 * s.size:
                        vis[labels_map.get(str(int(k)), "")] += 1
            landmarks = [k for k, _ in vis.most_common() if k and k != e.spec["target"]["type"]
                         and not k.startswith(("wall", "floor", "ceiling", "door", "window"))]
            row.update({"target_visible_end": int(pix[-1]), "target_visible_frac": round(float((pix > 20).mean()), 2),
                        "depth_min": round(float(np.nanmin(np.where(depth > 0, depth, np.nan))), 3),
                        "landmarks": landmarks[:5]})
        text = label.instruction_draft(e.spec, room_types, rooms_seq, landmarks)
        row["instruction_draft"] = text
        if renderer is not None:
            overlay_video(rgb, depth, boxes, prims, stop, text, out / f"ep{n:02d}.mp4")
        stats.append(row)
    timing["render"] = t_render
    ok = [s for s in stats if not s["failed"]]
    summary = {"house": args.house, "episodes": len(stats), "ok": len(ok),
               "fail_reasons": Counter(s["failed"] for s in stats if s["failed"]),
               "frames_total": sum(s["frames"] for s in ok), "timing_s": {k: round(v, 1) for k, v in timing.items()}}
    (out / "stats.json").write_text(json.dumps({"summary": summary, "episodes": stats}, indent=1, default=str))
    print(json.dumps(summary, default=str))
    for s in stats:
        print(json.dumps(s, default=str))


if __name__ == "__main__":
    main()
