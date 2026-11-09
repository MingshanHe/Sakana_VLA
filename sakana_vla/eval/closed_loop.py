"""Closed-loop evaluation in HyperDrone (env: sakana_hyperdrone) against a running policy_server.

The policy sees the onboard RGB + instruction + state every `--exec` frames (10 Hz frames), predicts an H-step chunk
of increments, which becomes world waypoints from the current pose; RAPTOR position control on L2F dynamics tracks
them (same lead / z-sag compensation as the expert). A chunk whose first step has stop > 0.5 ends the episode.

    PYTHONPATH=. python -m sakana_vla.eval.closed_loop --run runs/v0_nano_stageA --split val_unseen
Spawns the policy server (env sakana_vla2) as a child process on a node-local Unix socket.
Success: stopped within 1.0 m of the goal viewpoint with the target visible (>= 150 px), no collision.
"""
import argparse
import zlib
import json
from multiprocessing.connection import Client
from pathlib import Path

import numpy as np

from datagen import _jitfix  # noqa: F401
from datagen.expert_rollout import COLLISION_CLEARANCE, LEAD_STEPS, Z_SAG_COMP, quat_from_yaw, yaw_from_quat
from datagen.geometry import build_maps
from datagen.render import Renderer, target_bbox

AUTHKEY = b"sakana-vla"
SUCCESS_DIST, MIN_PIXELS = 1.0, 150
# RAPTOR lags its yaw setpoint (~0.2 rad at the expert's yaw rates). The expert's continuous reference never resets
# that lag, but re-anchoring each chunk at the measured yaw would discard it every replan -> systematic under-turn.
YAW_LEAD = 30  # oracle replay (reference anchor): yaw lead 0 -> SR 0.94, 30 -> SR 1.00, NE 0.06 m


def YAW_LEAD_SET(v):
    global YAW_LEAD
    YAW_LEAD = int(v)


def subset_dir(name):
    for base in ("data/vla_dataset", "data/vla_parts"):
        if Path(base, name).exists():
            return Path(base, name)
    raise FileNotFoundError(name)


def load_episodes(run, split, limit, seed):
    import sys
    sys.path.insert(0, ".")
    from sakana_vla.data.dataset import FrameCache, split_indices
    args = json.loads((Path(run) / "args.json").read_text())
    cache = FrameCache(args["cache"])
    tr, vs, vu = split_indices(cache, unseen_houses=tuple(args.get("unseen_houses", [])), seed=args["seed"])
    idx = {"val_seen": vs, "val_unseen": vu, "train": tr}[split]
    eps = sorted(set(cache.ep[idx].tolist()))
    rng = np.random.default_rng(seed)
    if limit and len(eps) > limit:
        eps = sorted(rng.choice(eps, limit, replace=False).tolist())
    extras = {}
    out = []
    for gi in eps:
        e = cache.episodes[gi]
        if e["subset"] not in extras:
            d = subset_dir(e["subset"])
            extras[e["subset"]] = {x["episode_index"]: x for x in map(json.loads, (d / "meta/episodes_extra.jsonl").read_text().splitlines())}
        x = extras[e["subset"]][e["episode_index"]]
        out.append({"house": x["house"], "start": x["start"], "goal": x["goal"], "target": x["target"],
                    "geodesic": x["geodesic"], "instruction": e["caption"], "expert_frames": e["length"],
                    "id": f"{e['subset']}/{e['episode_index']}", "offset": e["offset"], "length": e["length"]})
    return out


def state_vec(pos, quat, vel, angvel):
    yaw = yaw_from_quat(quat)
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([c * vel[:, 0] + s * vel[:, 1], -s * vel[:, 0] + c * vel[:, 1], vel[:, 2], angvel[:, 2], pos[:, 2]], 1)


def run_house(eps, sim, conn, exec_frames, max_frames, video_dir=None, fixed_noise=True, stop_consecutive=2, oracle=None,
              anchor="reference", min_stop_frame=0, record_dir=None):
    meta = json.loads(Path(f"data/scenes/{eps[0]['house']}.json").read_text())
    maps = build_maps(meta["glb"])
    g, clr = maps["grid"], maps["clearance"]
    n_pad = sim.num_drones
    renderer = Renderer(meta["glb"], batch=n_pad)
    n = len(eps)
    sim.reset()
    pos0 = np.zeros((n_pad, 3), np.float32)
    pos0[n:] = [0, 0, 50.0]
    yaw0 = np.zeros(n_pad)
    for i, e in enumerate(eps):
        pos0[i], yaw0[i] = e["start"]["position"], e["start"]["yaw"]
    sim.state["position"] = pos0
    sim.state["orientation"] = quat_from_yaw(yaw0).astype(np.float32)
    sim.state["linear_velocity"] = np.zeros((n_pad, 3), np.float32)
    sim.state["angular_velocity"] = np.zeros((n_pad, 3), np.float32)
    goal = np.array([e["goal"] for e in eps])
    done = np.zeros(n, bool)
    status = ["running"] * n
    path_len = np.zeros(n)
    min_goal = np.full(n, np.inf)
    min_target = np.full(n, np.inf)
    lo_t = np.array([e["target"]["aabb"][0] for e in eps])
    hi_t = np.array([e["target"]["aabb"][1] for e in eps])
    hold = np.c_[pos0, yaw0].astype(np.float32)          # setpoint for finished drones
    frames_video = [[] for _ in range(n)] if video_dir else None
    frame = 0
    stop_votes = np.zeros(n, int)
    cur_ref = np.c_[pos0, yaw0].astype(np.float64)   # continuous reference pose (expert-style), per drone
    rec = [{"rgb": [], "pos": [], "quat": [], "status": [], "decisions": []} for _ in range(n)] if record_dir else None
    seeds = [zlib.crc32(e["id"].encode()) for e in eps]
    last_pos = pos0[:n].astype(np.float64).copy()
    while not done.all() and frame < max_frames:
        P, Q = sim.state["position"], sim.state["orientation"]
        V, W = sim.state["linear_velocity"], sim.state["angular_velocity"]
        rgb, _, seg = renderer.render(P, Q)
        act = np.flatnonzero(~done)
        if video_dir:
            for i in act:
                frames_video[i].append(rgb[i])
        chunks = np.zeros((n_pad, 12, 5), np.float32)
        if oracle is not None:  # executor check: replay the expert's recorded chunks open-loop
            from sakana_vla.data.dataset import chunk_increments
            pred = np.stack([chunk_increments(oracle, eps[i]["offset"] + min(frame, eps[i]["length"] - 1), 12) for i in act])
        else:
            conn.send({"images": rgb[act], "instructions": [eps[i]["instruction"] for i in act],
                       "states": state_vec(P[act], Q[act], V[act], W[act]),
                       "seeds": [seeds[i] for i in act] if fixed_noise else None})
            pred = conn.recv()
        H = pred.shape[1]
        chunks = np.zeros((n_pad, H, 5), np.float32)
        chunks[act] = pred
        yaw = yaw_from_quat(Q)
        # stop decision on the first step of the chunk
        for i in act:
            if frame < min_stop_frame:  # diagnostic only: forbid stopping before this frame
                chunks[i, :, 4] = 0.0
            stop_votes[i] = stop_votes[i] + 1 if chunks[i, 0, 4] > 0.5 else 0
            if stop_votes[i] >= stop_consecutive:
                done[i], status[i] = True, "stopped"
                hold[i] = [*P[i], yaw[i]]
            elif chunks[i, 0, 4] > 0.5:
                chunks[i, :, :4] = 0.0  # tentative stop: hover in place until confirmed
        # chunk -> world waypoints (10 Hz) -> 100 Hz reference with lead
        cum = np.cumsum(chunks[:, :, :4], axis=1)
        c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
        # increments are in the measured yaw frame; anchor them on the continuous reference (or the measured pose)
        base = cur_ref if anchor == "reference" else np.c_[P, np.unwrap(yaw)]
        wp = np.stack([base[:, None, 0] + c * cum[:, :, 0] - s * cum[:, :, 1], base[:, None, 1] + s * cum[:, :, 0] + c * cum[:, :, 1],
                       base[:, None, 2] + cum[:, :, 2], base[:, None, 3] + cum[:, :, 3]], -1)
        wp = np.concatenate([base[:, None], wp], 1)                             # (n_pad, H+1, 4) at t = 0, 0.1, ...
        tq = np.arange(H * 10 + 1) / 10.0
        ref = np.stack([np.stack([np.interp(tq, np.arange(H + 1), wp[i, :, d]) for d in range(4)], -1)
                        for i in range(n_pad)])                                 # (n_pad, 10H+1, 4)
        for j in range(exec_frames * 10):
            sp = ref[:, min(j + LEAD_STEPS, ref.shape[1] - 1)].copy()
            sp[:, 3] = ref[:, min(j + YAW_LEAD, ref.shape[1] - 1), 3]
            sp[:, 2] += Z_SAG_COMP
            for i in range(n_pad):
                if i >= n or done[i]:
                    sp[i] = hold[i] if i < n else [0, 0, 50.0, 0]
                    if i < n:
                        sp[i, 2] += Z_SAG_COMP
            sp[:, 3] = (sp[:, 3] + np.pi) % (2 * np.pi) - np.pi
            sim.step(sp.astype(np.float32))
            Pj = sim.state["position"][:n].astype(np.float64)
            if record_dir is not None and j % 10 == 9:
                Pr, Qr = sim.state["position"], sim.state["orientation"]
                rgb_r, _, _ = renderer.render(Pr, Qr)
                for i in range(n):
                    if status[i] == "running" or not rec[i]["status"] or rec[i]["status"][-1] == "running":
                        rec[i]["rgb"].append(rgb_r[i]); rec[i]["pos"].append(Pr[i].copy()); rec[i]["quat"].append(Qr[i].copy())
                        rec[i]["status"].append(status[i])
            for i in np.flatnonzero(~done):
                ijk = g.to_index(Pj[i])
                if not g.valid(ijk)[0] or clr[tuple(ijk)] < COLLISION_CLEARANCE:
                    done[i], status[i] = True, "collision"
                    hold[i] = [*Pj[i], yaw[i]]
            path_len += np.where(status_mask(status, n), np.linalg.norm(Pj - last_pos, axis=1), 0.0)
            last_pos = Pj
            min_goal = np.minimum(min_goal, np.linalg.norm(Pj - goal, axis=1))
            min_target = np.minimum(min_target, np.linalg.norm(np.maximum(0, np.maximum(lo_t - Pj, Pj - hi_t)), axis=1))
        if record_dir is not None:
            for i in range(n):
                rec[i]["decisions"].append({"frame": frame, "wp": wp[i, :, :3].copy(), "stop0": float(chunks[i, 0, 4]),
                                            "active": bool(i in set(act.tolist()))})
        cur_ref = ref[:, exec_frames * 10].copy()
        frame += exec_frames
    for i in range(n):
        if status[i] == "running":
            status[i] = "timeout"
    P, Q = sim.state["position"], sim.state["orientation"]
    _, _, seg = renderer.render(P, Q)
    res = []
    for i, e in enumerate(eps):
        _, pix = target_bbox(seg[i:i + 1], e["target"]["segmentation_class"])
        dist = float(np.linalg.norm(P[i] - goal[i]))
        success = status[i] == "stopped" and dist <= SUCCESS_DIST and pix[0] >= MIN_PIXELS
        geo = e["geodesic"]
        d_box = float(np.linalg.norm(np.maximum(0, np.maximum(lo_t[i] - P[i], P[i] - hi_t[i]))))
        res.append({"id": e["id"], "house": e["house"], "target": e["target"]["type"], "status": status[i],
                    "near_target_end": bool(d_box <= 1.5), "near_target_ever": bool(min_target[i] <= 1.5),
                    "success": bool(success), "oracle_success": bool(min_goal[i] <= SUCCESS_DIST), "ne": dist,
                    "spl": float(success) * geo / max(path_len[i], geo), "path_len": float(path_len[i]),
                    "geodesic": geo, "target_pixels_end": int(pix[0]), "frames": frame})
        if video_dir and frames_video[i]:
            import imageio.v2 as imageio
            imageio.mimsave(Path(video_dir) / f"{e['id'].replace('/', '_')}_{status[i]}.mp4", frames_video[i], fps=10 / exec_frames,
                            macro_block_size=1)
    if record_dir is not None:
        Path(record_dir).mkdir(parents=True, exist_ok=True)
        for i, e in enumerate(eps):
            r = rec[i]
            np.savez_compressed(Path(record_dir) / f"{e['id'].replace('/', '_')}.npz", rgb=np.array(r["rgb"], np.uint8),
                                pos=np.array(r["pos"]), quat=np.array(r["quat"]), status=np.array(r["status"]),
                                dec_frame=np.array([d["frame"] for d in r["decisions"]]),
                                dec_wp=np.array([d["wp"] for d in r["decisions"]]),
                                dec_stop=np.array([d["stop0"] for d in r["decisions"]]),
                                dec_active=np.array([d["active"] for d in r["decisions"]]),
                                meta=json.dumps({"episode": e, "result": res[i]}))
    del renderer
    return res


def status_mask(status, n):
    return np.array([s in ("running", "stopped") for s in status[:n]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="val_unseen", choices=["val_seen", "val_unseen", "train"])
    ap.add_argument("--limit", type=int, default=64)
    ap.add_argument("--exec", type=int, default=5, help="frames executed per chunk before re-planning")
    ap.add_argument("--max-seconds", type=float, default=60.0)
    ap.add_argument("--server-python", default=str(Path.home() / "miniconda3/envs/sakana_vla2/bin/python"))
    ap.add_argument("--server-args", default="", help="extra policy_server args, e.g. '--blind' or '--shuffle-instructions'")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--videos", type=int, default=4, help="save onboard videos for the first N houses' episodes")
    ap.add_argument("--tag", default="")
    ap.add_argument("--resample-noise", action="store_true", help="fresh source noise every replan (default: fixed per episode)")
    ap.add_argument("--stop-consecutive", type=int, default=2)
    ap.add_argument("--oracle-replay", action="store_true", help="feed recorded expert chunks (executor sanity check)")
    ap.add_argument("--yaw-lead", type=int, default=YAW_LEAD, help="yaw setpoint lookahead (100 Hz steps)")
    ap.add_argument("--anchor", default="reference", choices=["reference", "measured"])
    ap.add_argument("--min-stop-frame", type=int, default=0, help="diagnostic: no stop before this frame (10 Hz)")
    ap.add_argument("--ids", nargs="*", default=None, help="only these episode ids (subset/episode_index)")
    ap.add_argument("--record", action="store_true", help="save 10 Hz onboard frames + trajectories for videos")
    args = ap.parse_args()
    YAW_LEAD_SET(args.yaw_lead)
    eps = load_episodes(args.run, args.split, 0 if args.ids else args.limit, seed=1)
    if args.ids:
        eps = [e for e in eps if e["id"] in set(args.ids)]
    from hyperdrone import dynamics
    sim = dynamics.Sim(args.batch, parameters={"dynamics": {"dynamics": {"model": "crazyflie"}}}, device="cpu",
                       control="position", seed=0)
    out_dir = Path(args.run) / f"closed_loop_{args.split}{('_' + args.tag) if args.tag else ''}"
    out_dir.mkdir(parents=True, exist_ok=True)
    import os, subprocess, tempfile, time
    oracle = None
    if args.oracle_replay:
        from sakana_vla.data.dataset import FrameCache
        oracle = FrameCache(json.loads((Path(args.run) / "args.json").read_text())["cache"])
    sock = os.path.join(tempfile.gettempdir(), f"sakana_vla_{os.getpid()}.sock")
    env = dict(os.environ, PYTHONNOUSERSITE="1", HF_HUB_DISABLE_PROGRESS_BARS="1")
    server = subprocess.Popen([args.server_python, "-u", "-m", "sakana_vla.eval.policy_server", "--run", args.run,
                               "--socket", sock, *args.server_args.split()], env=env,
                              stdout=open(out_dir / "policy_server.log", "w"), stderr=subprocess.STDOUT)
    while not os.path.exists(sock):
        if server.poll() is not None:
            raise RuntimeError(f"policy server exited; see {out_dir / 'policy_server.log'}")
        time.sleep(1)
    time.sleep(1)
    conn = Client(sock, family="AF_UNIX", authkey=AUTHKEY)
    by_house = {}
    for e in eps:
        by_house.setdefault(e["house"], []).append(e)
    results = []
    for h_i, (house, group) in enumerate(sorted(by_house.items())):
        for b in range(0, len(group), args.batch):
            vid = out_dir / "videos" if h_i < args.videos else None
            if vid:
                vid.mkdir(exist_ok=True)
            r = run_house(group[b:b + args.batch], sim, conn, args.exec, int(args.max_seconds * 10), vid,
                          fixed_noise=not args.resample_noise, stop_consecutive=args.stop_consecutive, oracle=oracle,
                          anchor=args.anchor, min_stop_frame=args.min_stop_frame,
                          record_dir=(out_dir / "record") if args.record else None)
            results += r
            sr = np.mean([x["success"] for x in r])
            print(f"{house}: {len(r)} eps, SR {sr:.2f}, status {dict(zip(*np.unique([x['status'] for x in r], return_counts=True)))}",
                  flush=True)
    conn.send("close")
    server.wait(timeout=60)
    summ = {k: float(np.mean([r[k] for r in results]))
            for k in ("success", "oracle_success", "ne", "spl", "near_target_end", "near_target_ever")}
    summ["collision_rate"] = float(np.mean([r["status"] == "collision" for r in results]))
    summ["timeout_rate"] = float(np.mean([r["status"] == "timeout" for r in results]))
    summ["n"] = len(results)
    (out_dir / "results.json").write_text(json.dumps({"summary": summ, "args": vars(args), "episodes": results}, indent=1))
    print("SUMMARY " + json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
