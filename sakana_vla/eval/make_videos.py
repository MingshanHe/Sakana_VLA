"""Compose result videos from closed_loop --record outputs (env: sakana_hyperdrone, CPU only).

    PYTHONPATH=. python -m sakana_vla.eval.make_videos --model runs/v1_stageA/closed_loop_val_unseen_video \
        --expert runs/v1_stageA/closed_loop_val_unseen_video_expert --out runs/v1_stageA/videos
Per episode: [model onboard view + projected chunk waypoints | expert-replay onboard view | top-down map], 10 fps.
Also writes all_episodes.mp4 (episodes back to back).
"""
import argparse
import json
import textwrap
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from datagen.geometry import build_maps, flight_mask
from datagen.render import FOV_DEG, camera_pose

S = 384          # onboard panel size
MAP_H = 384      # map panel height
STATUS_COLOR = {"stopped": (80, 200, 80), "collision": (60, 60, 230), "timeout": (0, 170, 230), "running": (200, 200, 200)}


def project(points, pos, quat, size=256):
    """World points -> pixel coords in the body-mounted camera (None if behind)."""
    eye, fwd, up = camera_pose(np.asarray(pos, float), np.asarray(quat, float))
    left = np.cross(up, fwd)
    f = (size / 2) / np.tan(np.deg2rad(FOV_DEG) / 2)
    out = []
    for p in points:
        d = np.asarray(p) - eye
        x, y, z = d @ fwd, d @ left, d @ up
        out.append(None if x < 0.05 else (size / 2 - f * y / x, size / 2 - f * z / x))
    return out


class TopDown:
    def __init__(self, meta, maps, mask, width):
        self.g = maps["grid"]
        m = mask.any(2)
        rows, cols = np.nonzero(m)
        pad = 8
        self.r0, self.r1 = max(0, rows.min() - pad), min(m.shape[0], rows.max() + pad)
        self.c0, self.c1 = max(0, cols.min() - pad), min(m.shape[1], cols.max() + pad)
        crop = m[self.r0:self.r1, self.c0:self.c1]
        self.scale = min(width / crop.shape[1], MAP_H / crop.shape[0])
        base = np.where(crop, 235, 70).astype(np.uint8)
        self.base = cv2.cvtColor(cv2.resize(base, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_NEAREST),
                                 cv2.COLOR_GRAY2BGR)

    def px(self, p):
        ij = (np.asarray(p)[..., :2] - self.g.origin[:2]) / self.g.res
        return np.stack([(ij[..., 1] - self.c0) * self.scale, (ij[..., 0] - self.r0) * self.scale], -1).astype(np.int32)


def load(path):
    d = np.load(path, allow_pickle=False)
    out = {k: d[k] for k in d.files if k != "meta"}
    meta = json.loads(str(d["meta"]))
    if "episode" in meta:                      # current format: {"episode": ..., "result": ...}
        meta = {**meta["result"], **{k: v for k, v in meta["episode"].items() if k != "target"},
                "target": meta["episode"]["target"]}
    elif isinstance(meta.get("target"), str):  # early recordings: result fields overwrote the target dict
        subset, ei = meta["id"].split("/")
        from .closed_loop import subset_dir
        for line in (subset_dir(subset) / "meta/episodes_extra.jsonl").read_text().splitlines():
            x = json.loads(line)
            if x["episode_index"] == int(ei):
                meta["target"] = x["target"]
                break
    out["meta"] = meta
    return out


def text_bar(lines, width, height, scale=0.5):
    bar = np.full((height, width, 3), 25, np.uint8)
    y = 18
    for ln, color in lines:
        cv2.putText(bar, ln, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
        y += int(22 * scale / 0.5)
    return bar


def compose_episode(mrec, erec, out_path):
    meta = mrec["meta"]
    scene = json.loads(Path(f"data/scenes/{meta['house']}.json").read_text())
    maps = build_maps(scene["glb"])
    td = TopDown(scene, maps, flight_mask(maps), width=S)
    T = max(len(mrec["rgb"]), len(erec["rgb"]) if erec else 0)
    lo, hi = np.array(meta["target"]["aabb"])
    goal = np.array(meta["goal"])
    exp_path = td.px(erec["pos"]) if erec else None
    mod_path = td.px(mrec["pos"])
    res_m, res_e = meta, (erec["meta"] if erec else None)
    head = [(f"{meta['id']}  ({meta['house']})   target: {meta['target']['type']}", (255, 255, 255))]
    for ln in textwrap.wrap("Instruction: " + meta["instruction"], 110)[:3]:
        head.append((ln, (200, 220, 255)))
    outcome = (f"MODEL: {res_m['status']}  success={res_m['success']}  final dist to goal {res_m['ne']:.2f} m  path {res_m['path_len']:.1f} m"
               + (f"   |   EXPERT REPLAY: {res_e['status']}  dist {res_e['ne']:.2f} m" if res_e else ""))
    head.append((outcome, (120, 255, 160) if res_m["success"] else (130, 170, 255)))
    W = 3 * S
    frames = []
    for t in range(T):
        tm = min(t, len(mrec["rgb"]) - 1)
        # --- model onboard with projected waypoints of the latest decision
        a = cv2.cvtColor(mrec["rgb"][tm], cv2.COLOR_RGB2BGR)
        a = cv2.resize(a, (S, S), interpolation=cv2.INTER_LINEAR)
        k = np.searchsorted(mrec["dec_frame"], tm, side="right") - 1
        stop_p = None
        if k >= 0 and mrec["dec_active"][k]:
            wp = mrec["dec_wp"][k][1:]
            stop_p = float(mrec["dec_stop"][k])
            pts = project(wp, mrec["pos"][tm], mrec["quat"][tm])
            for j, q in enumerate(pts):
                if q is not None and 0 <= q[0] < 256 and 0 <= q[1] < 256:
                    c = (0, int(255 - 15 * j), int(20 * j))
                    cv2.circle(a, (int(q[0] * S / 256), int(q[1] * S / 256)), 5, c, -1)
        st = str(mrec["status"][tm])
        cv2.putText(a, f"MODEL  t={tm / 10:4.1f}s  {st}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, STATUS_COLOR.get(st, (255, 255, 255)), 2, cv2.LINE_AA)
        if stop_p is not None:
            cv2.putText(a, f"stop p={stop_p:.2f}", (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1, cv2.LINE_AA)
        # --- expert replay onboard
        if erec is not None:
            te = min(t, len(erec["rgb"]) - 1)
            b = cv2.resize(cv2.cvtColor(erec["rgb"][te], cv2.COLOR_RGB2BGR), (S, S))
            cv2.putText(b, f"EXPERT REPLAY  t={te / 10:4.1f}s  {erec['status'][te]}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 200, 120), 2, cv2.LINE_AA)
        else:
            b = np.zeros((S, S, 3), np.uint8)
        # --- top-down
        m = td.base.copy()
        cv2.rectangle(m, tuple(td.px(lo)), tuple(td.px(hi)), (40, 40, 220), 2)
        cv2.drawMarker(m, tuple(td.px(goal)), (40, 40, 220), cv2.MARKER_STAR, 14, 2)
        cv2.circle(m, tuple(td.px(np.array(meta["start"]["position"]))), 5, (0, 0, 0), -1)
        if exp_path is not None:
            cv2.polylines(m, [exp_path[:min(t, len(exp_path) - 1) + 1]], False, (220, 140, 40), 2)
        cv2.polylines(m, [mod_path[:tm + 1]], False, (0, 140, 255), 2)
        p = mod_path[tm]
        yaw = np.arctan2(2 * (mrec["quat"][tm][0] * mrec["quat"][tm][3] + mrec["quat"][tm][1] * mrec["quat"][tm][2]),
                         1 - 2 * (mrec["quat"][tm][2] ** 2 + mrec["quat"][tm][3] ** 2))
        # map axes: x-world -> rows (down), y-world -> cols (right)
        tip = p + (np.array([np.sin(yaw), np.cos(yaw)]) * 18).astype(np.int32)
        cv2.arrowedLine(m, tuple(p), tuple(tip), (0, 90, 200), 2, tipLength=0.4)
        if k >= 0 and mrec["dec_active"][k]:
            for q in td.px(mrec["dec_wp"][k][1:]):
                cv2.circle(m, tuple(q), 2, (0, 200, 0), -1)
        mp = np.full((S, S, 3), 70, np.uint8)
        mp[:min(S, m.shape[0]), :min(S, m.shape[1])] = m[:S, :S]
        cv2.putText(mp, "orange: model   blue: expert   red: target/goal", (6, S - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (20, 20, 20), 1, cv2.LINE_AA)
        frame = np.concatenate([text_bar(head, W, 112), np.concatenate([a, b, mp], 1)], 0)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    frames += [frames[-1]] * 15
    imageio.mimsave(out_path, frames, fps=10, macro_block_size=1, quality=7)
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--expert", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    allf = []
    for f in sorted(Path(args.model, "record").glob("*.npz")):
        mrec = load(f)
        ef = Path(args.expert, "record", f.name) if args.expert else None
        erec = load(ef) if ef and ef.exists() else None
        name = f"{f.stem}_{mrec['meta']['status']}.mp4"
        fr = compose_episode(mrec, erec, out / name)
        allf += fr
        print(f"wrote {out / name} ({len(fr)} frames)", flush=True)
    imageio.mimsave(out / "all_episodes.mp4", allf, fps=10, macro_block_size=1, quality=7)
    print(f"wrote {out / 'all_episodes.mp4'}")


if __name__ == "__main__":
    main()
