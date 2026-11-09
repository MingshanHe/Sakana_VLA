"""LeRobot v2.1 writer, field names aligned with OpenFly v2 / AeroBrain plus indoor extras.

<root>/<subset>/
  meta/info.json, tasks.jsonl, episodes.jsonl, episodes_stats.jsonl
  meta/episodes_extra.jsonl        structured per-episode metadata (target, rooms, segments, landmarks, ...)
  data/chunk-XXX/episode_XXXXXX.parquet
  videos/chunk-XXX/observation.images.rgb/episode_XXXXXX.mp4    AV1 (libsvtav1), yuv420p
  depth/chunk-XXX/episode_XXXXXX.npz                             uint16 millimetres, key "depth" (T,H,W)
  instructions/raw.jsonl           template drafts (LLM rewrite + human check go to llm_v1 / verified_v1)
  keyframes/episode_XXXXXX/*.jpg   start / segment boundaries / end, for annotation
"""
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .label import PRIMITIVES

FPS = 10
CHUNK = 1000
RGB_KEY = "observation.images.rgb"


def ffmpeg_exe():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def encode_av1(frames, path, fps=FPS, crf=30, gop=2):
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames.shape[1:3]
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "-", "-c:v", "libsvtav1", "-pix_fmt", "yuv420p", "-g", str(gop), "-crf", str(crf),
           "-svtav1-params", "lp=2", str(path)]
    subprocess.run(cmd, input=np.ascontiguousarray(frames, np.uint8).tobytes(), check=True)


def _stats(x):
    x = np.asarray(x, np.float64)
    if x.ndim == 1:
        x = x[:, None]
    return {"min": x.min(0).tolist(), "max": x.max(0).tolist(), "mean": x.mean(0).tolist(),
            "std": x.std(0).tolist(), "count": [len(x)]}


def _image_stats(frames):
    f = frames.astype(np.float64) / 255.0
    per = lambda fn: [[[float(v)]] for v in fn(f, axis=(0, 1, 2))]
    return {"min": per(np.min), "max": per(np.max), "mean": per(np.mean), "std": per(np.std), "count": [len(frames)]}


def features(h, w):
    vec = lambda n, names: {"dtype": "float32", "shape": [n], "names": {"motors": names}}
    scalar = lambda dt: {"dtype": dt, "shape": [1], "names": None}
    return {
        RGB_KEY: {"dtype": "video", "shape": [h, w, 3], "names": ["height", "width", "rgb"],
                  "info": {"video.height": h, "video.width": w, "video.codec": "av1", "video.pix_fmt": "yuv420p",
                           "video.is_depth_map": False, "video.fps": FPS, "video.channels": 3, "has_audio": False}},
        "observation.states": vec(4, ["x", "y", "z", "yaw"]),
        "observation.state_full": vec(13, ["px", "py", "pz", "qw", "qx", "qy", "qz", "vx", "vy", "vz", "wx", "wy", "wz"]),
        "observation.boxes": vec(4, ["x", "y", "w", "h"]),
        "observation.target_pixels": scalar("int64"),
        "actions": vec(5, ["x", "y", "z", "yaw", "s"]),
        "actions_aerobrain": vec(4, ["x", "y", "z", "s"]),
        "caption": {"dtype": "string", "shape": [1]},
        "env": {"dtype": "string", "shape": [1]},
        "primitive": {"dtype": "string", "shape": [1]},
        "phase": {"dtype": "string", "shape": [1]},
        "perturbed": scalar("bool"),
        "timestamp": scalar("float32"), "frame_index": scalar("int64"), "episode_index": scalar("int64"),
        "index": scalar("int64"), "task_index": scalar("int64"),
    }


class DatasetWriter:
    def __init__(self, root, subset, h, w):
        self.dir = Path(root) / subset
        (self.dir / "meta").mkdir(parents=True, exist_ok=True)
        self.h, self.w = h, w
        self.episodes, self.stats, self.extra, self.instr = [], [], [], []
        self.total_frames = 0

    def add(self, ep_index, arr, acts, stop, prims, boxes, pix, rgb, depth_mm, spec, extra, instruction, keyframes):
        n = len(acts)
        pos, yaw = arr["position"], arr["yaw"]
        c0, s0 = np.cos(yaw[0]), np.sin(yaw[0])
        d = pos - pos[0]
        states = np.c_[c0 * d[:, 0] + s0 * d[:, 1], -s0 * d[:, 0] + c0 * d[:, 1], d[:, 2], yaw - yaw[0]].astype(np.float32)
        full = np.c_[pos, arr["orientation"], arr["linear_velocity"], arr["angular_velocity"]].astype(np.float32)
        acts_ab = np.c_[acts[:, :3], acts[:, 4:5]].astype(np.float32)
        chunk = ep_index // CHUNK
        table = pa.table({
            "observation.states": list(states), "observation.state_full": list(full),
            "observation.boxes": list(boxes.astype(np.float32)), "observation.target_pixels": pix.astype(np.int64),
            "actions": list(acts), "actions_aerobrain": list(acts_ab),
            "caption": [instruction] * n, "env": [spec["house"]] * n, "primitive": prims,
            "phase": arr["phase"], "perturbed": arr["perturbed"].astype(bool),
            "timestamp": (np.arange(n) / FPS).astype(np.float32), "frame_index": np.arange(n, dtype=np.int64),
            "episode_index": np.full(n, ep_index, np.int64),
            "index": np.arange(self.total_frames, self.total_frames + n, dtype=np.int64),
            "task_index": np.array([PRIMITIVES.index(p) for p in prims], np.int64),
        })
        p = self.dir / f"data/chunk-{chunk:03d}/episode_{ep_index:06d}.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, p)
        encode_av1(rgb, self.dir / f"videos/chunk-{chunk:03d}/{RGB_KEY}/episode_{ep_index:06d}.mp4")
        dp = self.dir / f"depth/chunk-{chunk:03d}/episode_{ep_index:06d}.npz"
        dp.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dp, depth=depth_mm)
        kd = self.dir / f"keyframes/episode_{ep_index:06d}"
        kd.mkdir(parents=True, exist_ok=True)
        for name, t in keyframes:
            cv2.imwrite(str(kd / f"{t:04d}_{name}.jpg"), cv2.cvtColor(rgb[t], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])

        tasks = sorted(set(prims), key=prims.index)
        self.episodes.append({"episode_index": ep_index, "tasks": tasks, "length": n})
        self.stats.append({"episode_index": ep_index, "stats": {
            RGB_KEY: _image_stats(rgb), "observation.states": _stats(states), "observation.state_full": _stats(full),
            "observation.boxes": _stats(boxes), "actions": _stats(acts), "actions_aerobrain": _stats(acts_ab),
            "timestamp": _stats(np.arange(n) / FPS), "frame_index": _stats(np.arange(n)),
            "episode_index": _stats(np.full(n, ep_index)), "task_index": _stats(table["task_index"].to_numpy())}})
        self.extra.append({"episode_index": ep_index, **extra})
        self.instr.append({"episode_index": ep_index, "version": "raw", "instruction": instruction})
        self.total_frames += n

    def close(self):
        m = self.dir / "meta"
        n_ep = len(self.episodes)
        info = {"codebase_version": "v2.1", "robot_type": "crazyflie_l2f_raptor", "total_episodes": n_ep,
                "total_frames": self.total_frames, "total_tasks": len(PRIMITIVES), "total_videos": n_ep,
                "total_chunks": (max((e["episode_index"] for e in self.episodes), default=0) // CHUNK) + 1,
                "chunks_size": CHUNK, "fps": FPS, "splits": {"train": f"0:{n_ep}"},
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
                "features": features(self.h, self.w)}
        (m / "info.json").write_text(json.dumps(info, indent=4))
        dump = lambda name, rows: (m / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
        dump("tasks.jsonl", [{"task_index": i, "task": t} for i, t in enumerate(PRIMITIVES)])
        dump("episodes.jsonl", self.episodes)
        dump("episodes_stats.jsonl", self.stats)
        dump("episodes_extra.jsonl", self.extra)
        (self.dir / "instructions").mkdir(exist_ok=True)
        (self.dir / "instructions/raw.jsonl").write_text("".join(json.dumps(r) + "\n" for r in self.instr))
