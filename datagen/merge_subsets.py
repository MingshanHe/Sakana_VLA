"""Merge per-house parts (written by generate.py in parallel jobs) into one LeRobot v2.1 subset.

    PYTHONPATH=. python -m datagen.merge_subsets --parts data/vla_parts/indoor_nano_h* --out data/vla_dataset/indoor_nano
Episodes are renumbered contiguously (episode_index, index, file names); meta/instructions are concatenated.
"""
import argparse
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .write_dataset import CHUNK, RGB_KEY


def jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    (out / "meta").mkdir(parents=True)
    (out / "instructions").mkdir()
    episodes, stats, extra, instr, reports = [], [], [], [], []
    ep_new, frame_new, info = 0, 0, None
    for part in sorted(Path(p) for p in args.parts):
        info = json.loads((part / "meta/info.json").read_text())
        p_eps = jsonl(part / "meta/episodes.jsonl")
        p_stats = {d["episode_index"]: d for d in jsonl(part / "meta/episodes_stats.jsonl")}
        p_extra = {d["episode_index"]: d for d in jsonl(part / "meta/episodes_extra.jsonl")}
        p_instr = {d["episode_index"]: d for d in jsonl(part / "instructions/raw.jsonl")}
        rep = part / "meta/generation_report.json"
        if rep.exists():
            reports.append({"part": part.name, **json.loads(rep.read_text())})
        for e in p_eps:
            old, n = e["episode_index"], e["length"]
            oc, nc = old // CHUNK, ep_new // CHUNK
            t = pq.read_table(part / f"data/chunk-{oc:03d}/episode_{old:06d}.parquet")
            t = t.set_column(t.schema.get_field_index("episode_index"), "episode_index",
                             pa.array([ep_new] * n, pa.int64()))
            t = t.set_column(t.schema.get_field_index("index"), "index",
                             pa.array(range(frame_new, frame_new + n), pa.int64()))
            dst = out / f"data/chunk-{nc:03d}/episode_{ep_new:06d}.parquet"
            dst.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(t, dst)
            for src, dst in [
                (part / f"videos/chunk-{oc:03d}/{RGB_KEY}/episode_{old:06d}.mp4",
                 out / f"videos/chunk-{nc:03d}/{RGB_KEY}/episode_{ep_new:06d}.mp4"),
                (part / f"depth/chunk-{oc:03d}/episode_{old:06d}.npz",
                 out / f"depth/chunk-{nc:03d}/episode_{ep_new:06d}.npz")]:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            shutil.copytree(part / f"keyframes/episode_{old:06d}", out / f"keyframes/episode_{ep_new:06d}")
            episodes.append({**e, "episode_index": ep_new})
            st = p_stats[old]
            st["stats"]["episode_index"] = {"min": [ep_new], "max": [ep_new], "mean": [float(ep_new)], "std": [0.0],
                                            "count": [n]}
            stats.append({**st, "episode_index": ep_new})
            extra.append({**p_extra[old], "episode_index": ep_new, "source_part": part.name, "source_episode": old})
            instr.append({**p_instr[old], "episode_index": ep_new})
            ep_new += 1
            frame_new += n
    info.update({"total_episodes": ep_new, "total_frames": frame_new, "total_videos": ep_new,
                 "total_chunks": (ep_new - 1) // CHUNK + 1, "splits": {"train": f"0:{ep_new}"}})
    (out / "meta/info.json").write_text(json.dumps(info, indent=4))
    dump = lambda p, rows: p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    shutil.copy2(Path(args.parts[0]) / "meta/tasks.jsonl", out / "meta/tasks.jsonl")
    dump(out / "meta/episodes.jsonl", episodes)
    dump(out / "meta/episodes_stats.jsonl", stats)
    dump(out / "meta/episodes_extra.jsonl", extra)
    dump(out / "instructions/raw.jsonl", instr)
    (out / "meta/generation_report.json").write_text(json.dumps(reports, indent=1))
    print(f"merged {len(args.parts)} parts -> {out}: {ep_new} episodes, {frame_new} frames")


if __name__ == "__main__":
    main()
