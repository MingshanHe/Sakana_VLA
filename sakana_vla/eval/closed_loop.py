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
from datagen.geometry import build_maps, flight_mask
from datagen.planner import GoalField
from datagen.render import Renderer, target_bbox
from sakana_vla.wm.labels import GOAL_DIST, VIS_PIXELS, clearance_at, rgbd64

AUTHKEY = b"sakana-vla"
SUCCESS_DIST, MIN_PIXELS = 1.0, 150
ORACLE_PENALTY = 10.0  # privileged candidate scoring: collision / wrong-stop penalty, correct-stop bonus
SHIELD_CONT, SHIELD_CLR = 0.7, 0.2  # wm_shield: a candidate is unsafe if P(no collision) < 0.7 or predicted min clearance < 0.2 m


def chunk_waypoints(chunk, base, yaw):
    """(H, 5) increments in the measured-yaw frame -> (H+1, 3) world points starting at base[:3]."""
    cum = np.cumsum(chunk[:, :3], 0)
    c, s = np.cos(yaw), np.sin(yaw)
    pts = np.c_[base[0] + c * cum[:, 0] - s * cum[:, 1], base[1] + s * cum[:, 0] + c * cum[:, 1], base[2] + cum[:, 2]]
    return np.concatenate([np.asarray(base[:3], float)[None], pts], 0)


def path_min_clearance(pts, g, clr):
    seg = np.concatenate([pts[:-1], (pts[:-1] + pts[1:]) / 2, pts[1:]], 0)
    ijk = g.to_index(seg)
    ok = g.valid(ijk)
    if not np.all(ok):
        return 0.0
    return float(clr[ijk[:, 0], ijk[:, 1], ijk[:, 2]].min())


def score_candidates(cands, pos, yaw, base, goal, field, g, clr, at_goal):
    """Privileged (oracle) score per candidate chunk: geodesic progress to the goal over the chunk, minus a collision
    penalty and a soft clearance penalty. A stop candidate scores +penalty if the drone is at the goal with the target
    visible, else -penalty. This is the best-of-N upper bound a learned verifier would have to approach."""
    scores = np.zeros(len(cands))
    geo_now = field.geodesic(pos) if field is not None else float(np.linalg.norm(pos - goal))
    for k, ch in enumerate(cands):
        if ch[0, 4] > 0.5:
            scores[k] = ORACLE_PENALTY if at_goal else -ORACLE_PENALTY
            continue
        pts = chunk_waypoints(ch, base, yaw)
        min_clr = path_min_clearance(pts, g, clr)
        geo_end = field.geodesic(pts[-1]) if field is not None else float(np.linalg.norm(pts[-1] - goal))
        scores[k] = ((geo_now - geo_end) - (ORACLE_PENALTY if min_clr < COLLISION_CLEARANCE else 0.0)
                     - 0.5 * max(0.0, 0.3 - min_clr) - (1.0 if at_goal else 0.0))
    return scores
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
              anchor="reference", min_stop_frame=0, record_dir=None, candidates=1, select="first", tentative_stop="hover",
              dump_wm=None, primitives=False):
    meta = json.loads(Path(f"data/scenes/{eps[0]['house']}.json").read_text())
    maps = build_maps(meta["glb"])
    g, clr = maps["grid"], maps["clearance"]
    fields = None
    if select.startswith(("oracle", "wm")) or dump_wm is not None:  # shortest-path field per goal (oracle scoring, dump labels)
        mask = flight_mask(maps)
        fields = []
        for e in eps:
            try:
                fields.append(GoalField(maps, mask, g.to_index(np.asarray(e["goal"]))))
            except ValueError:
                fields.append(None)
    rng_sel = np.random.default_rng(0)
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
    end_frame = np.full(n, np.nan)
    n_dec, n_changed, n_stop_rej, n_stop_forced = (np.zeros(n, int) for _ in range(4))
    n_ver_dec, n_ver_top1, ver_regret = np.zeros(n, int), np.zeros(n, int), np.zeros(n)  # learned verifier vs oracle
    n_prim_pick = np.zeros(n, int)  # decisions where a primitive candidate (index >= candidates) was executed
    prev_wp = None
    need_wm = select.startswith("wm") or dump_wm is not None   # send RGB-D + executed increments to the server / dump them
    last_exec = np.zeros((n_pad, 25), np.float32)              # increments executed since the previous decision
    clr_step_min = np.full(n, np.inf)                           # min clearance since the previous decision
    dump = [[] for _ in range(n)] if dump_wm is not None else None
    pending_term = []
    wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi
    while not done.all() and frame < max_frames:
        P, Q = sim.state["position"], sim.state["orientation"]
        V, W = sim.state["linear_velocity"], sim.state["angular_velocity"]
        rgb, depth, seg = renderer.render(P, Q)
        act = np.flatnonzero(~done)
        yaw = yaw_from_quat(Q)
        tentative = np.zeros(n_pad, bool)
        wm_obs = wm_scores = wm_goal = wm_parts = None
        if need_wm:
            wm_obs = np.stack([rgbd64(rgb[i], depth[i]) for i in act]) if len(act) else np.zeros((0, 64, 64, 4), np.uint8)

        def make_row(i, terminal):
            _, pix_i = target_bbox(seg[i:i + 1], eps[i]["target"]["segmentation_class"])
            vis = bool(pix_i[0] >= VIS_PIXELS)
            geo = fields[i].geodesic(P[i]) if fields[i] is not None else float(np.linalg.norm(P[i] - goal[i]))
            c0 = clr_step_min[i] if np.isfinite(clr_step_min[i]) else float(clearance_at(maps, P[i])[0])
            clr_step_min[i] = np.inf
            return {"image": rgbd64(rgb[i], depth[i]), "vector": state_vec(P[i:i + 1], Q[i:i + 1], V[i:i + 1], W[i:i + 1])[0],
                    "action": np.zeros(25, np.float32), "geodesic": geo, "clearance": float(c0), "target_vis": vis,
                    "at_goal": bool(np.linalg.norm(P[i] - goal[i]) <= GOAL_DIST and vis), "is_terminal": terminal}

        row_ref = {}
        if dump is not None:
            for i in pending_term:
                dump[i].append(make_row(i, True))
            pending_term = []
            for i in act:
                row_ref[i] = make_row(i, False)
                dump[i].append(row_ref[i])
        if video_dir:
            for i in act:
                frames_video[i].append(rgb[i])
        chunks = np.zeros((n_pad, 12, 5), np.float32)
        if oracle is not None:  # executor check: replay the expert's recorded chunks open-loop
            from sakana_vla.data.dataset import chunk_increments
            pred = np.stack([chunk_increments(oracle, eps[i]["offset"] + min(frame, eps[i]["length"] - 1), 12) for i in act])
        else:
            req = {"images": rgb[act], "instructions": [eps[i]["instruction"] for i in act],
                   "states": state_vec(P[act], Q[act], V[act], W[act]),
                   "seeds": [seeds[i] for i in act] if fixed_noise else None, "n_samples": candidates, "primitives": primitives}
            if need_wm:
                req["wm"] = {"ids": [eps[i]["id"] for i in act], "obs": wm_obs, "prev_actions": last_exec[act],
                             "is_first": np.array([frame == 0] * len(act), bool)}
            conn.send(req)
            resp = conn.recv()
            if isinstance(resp, dict):
                pred, wm_scores, wm_goal, wm_parts = resp["chunks"], resp.get("wm_scores"), resp.get("wm_goal"), resp.get("wm_parts")
                if select.startswith("wm") and pred.ndim == 4 and wm_scores is None:
                    raise RuntimeError("--select wm needs the policy server started with --wm <checkpoint> (use --server-args)")
            else:
                pred = resp
            if pred.ndim == 4:  # (n_act, N, H, 5): pick one chunk per drone
                picked = np.zeros((len(act),) + pred.shape[2:], np.float32)
                for a_i, i in enumerate(act):
                    cands = pred[a_i]
                    k = 0
                    if select == "random":
                        k = int(rng_sel.integers(len(cands)))
                    elif select.startswith("oracle") or select.startswith("wm"):
                        _, pix_now = target_bbox(seg[i:i + 1], eps[i]["target"]["segmentation_class"])
                        at_goal = bool(np.linalg.norm(P[i] - goal[i]) <= SUCCESS_DIST and pix_now[0] >= MIN_PIXELS)
                        base_i = cur_ref[i] if anchor == "reference" else np.r_[P[i], yaw[i]]
                        o_scores = score_candidates(cands, P[i], yaw[i], base_i, goal[i], fields[i], g, clr, at_goal)
                        if select.startswith("wm"):  # learned verifier picks; the oracle only grades its pick
                            scores = np.asarray(wm_scores[a_i], np.float64)
                            k_or, k_wm = int(np.argmax(o_scores)), int(np.argmax(scores))
                            n_ver_dec[i] += 1
                            n_ver_top1[i] += int(o_scores[k_wm] >= o_scores[k_or] - 1e-6)
                            ver_regret[i] += o_scores[k_or] - o_scores[k_wm]
                        else:
                            scores = o_scores
                        moving = np.flatnonzero(cands[:, 0, 4] <= 0.5)
                        if select.endswith("_motion"):  # the policy's own stop decision (candidate 0), scored motion
                            k = 0 if (cands[0, 0, 4] > 0.5 or len(moving) == 0) else int(moving[np.argmax(scores[moving])])
                        elif select == "wm_shield":  # keep the policy's choice unless the model predicts danger
                            unsafe = (wm_parts["cont"][a_i] < SHIELD_CONT) | (wm_parts["clr_min"][a_i] < SHIELD_CLR)
                            if cands[0, 0, 4] > 0.5 or not unsafe[0]:
                                k = 0
                            else:
                                safe = np.flatnonzero(~unsafe & (cands[:, 0, 4] <= 0.5))
                                k = int(safe[np.argmax(scores[safe])]) if len(safe) else int(np.argmax(scores))
                        elif select == "wm":  # the world model can veto the policy's stop, never force one
                            if cands[0, 0, 4] > 0.5 and (wm_goal is None or wm_goal[a_i] > 0.5 or len(moving) == 0):
                                k = 0
                            else:
                                k = int(moving[np.argmax(scores[moving])]) if len(moving) else 0
                        else:
                            k = int(np.argmax(scores))
                        n_stop_rej[i] += int(cands[0, 0, 4] > 0.5 and cands[k, 0, 4] <= 0.5)
                        n_stop_forced[i] += int(cands[0, 0, 4] <= 0.5 and cands[k, 0, 4] > 0.5)
                    n_dec[i] += 1
                    n_changed[i] += int(k != 0)
                    n_prim_pick[i] += int(primitives and k >= candidates)
                    picked[a_i] = cands[k]
                pred = picked
        H = pred.shape[1]
        chunks = np.zeros((n_pad, H, 5), np.float32)
        chunks[act] = pred
        # stop decision on the first step of the chunk
        for i in act:
            if frame < min_stop_frame:  # diagnostic only: forbid stopping before this frame
                chunks[i, :, 4] = 0.0
            stop_votes[i] = stop_votes[i] + 1 if chunks[i, 0, 4] > 0.5 else 0
            if stop_votes[i] >= stop_consecutive:
                done[i], status[i] = True, "stopped"
                hold[i] = [*P[i], yaw[i]]
                end_frame[i] = frame
            elif chunks[i, 0, 4] > 0.5:
                chunks[i, :, :4] = 0.0  # tentative stop: hover in place until confirmed (or continue, see below)
                tentative[i] = True
        # chunk -> world waypoints (10 Hz) -> 100 Hz reference with lead
        cum = np.cumsum(chunks[:, :, :4], axis=1)
        c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
        # increments are in the measured yaw frame; anchor them on the continuous reference (or the measured pose)
        base = cur_ref if anchor == "reference" else np.c_[P, np.unwrap(yaw)]
        wp = np.stack([base[:, None, 0] + c * cum[:, :, 0] - s * cum[:, :, 1], base[:, None, 1] + s * cum[:, :, 0] + c * cum[:, :, 1],
                       base[:, None, 2] + cum[:, :, 2], base[:, None, 3] + cum[:, :, 3]], -1)
        wp = np.concatenate([base[:, None], wp], 1)                             # (n_pad, H+1, 4) at t = 0, 0.1, ...
        if tentative_stop == "continue" and prev_wp is not None:
            for i in np.flatnonzero(tentative):  # keep flying the previous chunk's remainder instead of hovering
                tail = prev_wp[i, exec_frames:]
                wp[i] = np.concatenate([tail, np.repeat(tail[-1:], H + 1 - len(tail), 0)], 0)
        prev_wp = wp.copy()
        if need_wm:  # commanded increments for the next 0.5 s, in the measured-yaw frame of this decision
            dxyz = wp[:, 1:exec_frames + 1, :3] - wp[:, :exec_frames, :3]
            cy, sy = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
            inc = np.stack([cy * dxyz[..., 0] + sy * dxyz[..., 1], -sy * dxyz[..., 0] + cy * dxyz[..., 1], dxyz[..., 2],
                            wrap(wp[:, 1:exec_frames + 1, 3] - wp[:, :exec_frames, 3]), chunks[:, :exec_frames, 4]], -1)
            last_exec = inc.reshape(n_pad, -1).astype(np.float32)
            for i, row in row_ref.items():
                row["action"] = last_exec[i].copy()
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
            if dump is not None:
                clr_step_min = np.minimum(clr_step_min, clearance_at(maps, Pj))
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
                    end_frame[i] = frame + (j + 1) / 10
                    if dump is not None:
                        pending_term.append(i)
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
            end_frame[i] = frame
    P, Q = sim.state["position"], sim.state["orientation"]
    V, W = sim.state["linear_velocity"], sim.state["angular_velocity"]
    rgb, depth, seg = renderer.render(P, Q)
    if dump is not None:
        for i in pending_term:
            dump[i].append(make_row(i, True))
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
                    "geodesic": geo, "target_pixels_end": int(pix[0]), "frames": float(end_frame[i]),
                    "decisions": int(n_dec[i]), "sel_changed": int(n_changed[i]), "stop_rejected": int(n_stop_rej[i]),
                    "stop_forced": int(n_stop_forced[i]), "ver_decisions": int(n_ver_dec[i]), "ver_top1": int(n_ver_top1[i]),
                    "ver_regret": float(ver_regret[i]), "primitive_picks": int(n_prim_pick[i])})
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
    if dump is not None:
        Path(dump_wm).mkdir(parents=True, exist_ok=True)
        for i, e in enumerate(eps):
            rows = dump[i]
            if len(rows) < 2:
                continue
            geo = np.array([r["geodesic"] for r in rows], np.float32)
            reward = np.zeros(len(rows), np.float32)
            reward[1:] = geo[:-1] - geo[1:]
            is_first = np.zeros(len(rows), bool)
            is_first[0] = True
            np.savez_compressed(Path(dump_wm) / f"{e['id'].replace('/', '_')}.npz",
                                image=np.stack([r["image"] for r in rows]), vector=np.stack([r["vector"] for r in rows]).astype(np.float32),
                                action=np.stack([r["action"] for r in rows]), reward=reward,
                                clearance=np.array([r["clearance"] for r in rows], np.float32),
                                target_vis=np.array([r["target_vis"] for r in rows]), at_goal=np.array([r["at_goal"] for r in rows]),
                                stop=np.zeros(len(rows), bool), is_first=is_first,
                                is_terminal=np.array([r["is_terminal"] for r in rows]), geodesic=geo,
                                meta=json.dumps({"subset": e["id"].split("/")[0], "episode_index": int(e["id"].split("/")[1]),
                                                 "house": e["house"], "instruction": e["instruction"], "source": "policy",
                                                 "status": status[i]}))
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
    ap.add_argument("--candidates", type=int, default=1, help="chunks sampled per decision (different fixed noise each)")
    ap.add_argument("--select", default="first", choices=["first", "random", "oracle", "oracle_motion", "wm", "wm_motion", "wm_force", "wm_shield"],
                    help="pick among candidates; oracle* use privileged sim info (upper bound), wm* the learned world model "
                         "(wm: motion by score, stop only if the policy proposes it and the model agrees; wm_force: pure argmax)")
    ap.add_argument("--dump-wm", default=None, help="directory: save every episode as world-model replay (RGB-D, actions, labels)")
    ap.add_argument("--primitives", action="store_true", help="add 4 hand-made candidates (hover, rotate L/R, slow forward)")
    ap.add_argument("--tentative-stop", default="hover", choices=["hover", "continue"],
                    help="on the first stop vote: hover in place (v0) or keep flying the previous chunk")
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
                          record_dir=(out_dir / "record") if args.record else None, candidates=args.candidates,
                          select=args.select, tentative_stop=args.tentative_stop, dump_wm=args.dump_wm,
                          primitives=args.primitives)
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
    summ["stopped_rate"] = float(np.mean([r["status"] == "stopped" for r in results]))
    n_dec = sum(r["decisions"] for r in results)
    summ["sel_changed_rate"] = float(sum(r["sel_changed"] for r in results) / max(1, n_dec))
    summ["primitive_pick_rate"] = float(sum(r.get("primitive_picks", 0) for r in results) / max(1, n_dec))
    summ["stop_rejected"] = int(sum(r["stop_rejected"] for r in results))
    summ["stop_forced"] = int(sum(r["stop_forced"] for r in results))
    n_ver = sum(r["ver_decisions"] for r in results)
    if n_ver:
        summ["verifier_top1_acc"] = float(sum(r["ver_top1"] for r in results) / n_ver)
        summ["verifier_regret"] = float(sum(r["ver_regret"] for r in results) / n_ver)
    summ["n"] = len(results)
    (out_dir / "results.json").write_text(json.dumps({"summary": summ, "args": vars(args), "episodes": results}, indent=1))
    print("SUMMARY " + json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
