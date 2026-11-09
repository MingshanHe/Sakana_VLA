"""Per-frame labels from the realized 10 Hz trajectory: actions, stop, motion primitives, instruction draft.

actions[t] = [dx, dy, dz, dyaw, stop]: displacement p[t+1]-p[t] rotated into the yaw frame at t (z unchanged),
wrapped yaw change, stop flag. Chunks (OpenFly-v2 style) are assembled in the dataloader from
observation.states, so the chunk horizon can change without regenerating data.
"""
import numpy as np

from .expert_rollout import yaw_from_quat
from .planner import wrap

STOP_POS_TOL, STOP_YAW_TOL = 0.15, np.deg2rad(10)
PRIMITIVES = ["move forward", "slightly turn left", "turn left", "sharp turn left", "slightly turn right",
              "turn right", "sharp turn right", "rotate left", "rotate right", "ascend", "descend", "hover", "stop"]


def frame_arrays(records):
    pos = np.array([r["position"] for r in records], np.float64)
    quat = np.array([r["orientation"] for r in records], np.float64)
    yaw = np.unwrap(yaw_from_quat(quat))
    return {"position": pos, "orientation": quat, "yaw": yaw,
            "linear_velocity": np.array([r["linear_velocity"] for r in records], np.float64),
            "angular_velocity": np.array([r["angular_velocity"] for r in records], np.float64),
            "phase": [r["phase"] for r in records], "perturbed": np.array([r["perturbed"] for r in records])}


def actions_from_states(pos, yaw, stop):
    n = len(pos)
    d = np.zeros((n, 3))
    d[:-1] = pos[1:] - pos[:-1]
    c, s = np.cos(yaw), np.sin(yaw)
    body = np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1], d[:, 2]], axis=1)
    dyaw = np.zeros(n)
    dyaw[:-1] = wrap(np.diff(yaw))
    act = np.c_[body, dyaw, stop.astype(np.float64)]
    act[stop.astype(bool), :4] = 0.0  # AeroBrain convention: stop frames carry zero motion
    return act.astype(np.float32)


def stop_flags(arr, goal, target_center):
    yaw_goal = np.arctan2(target_center[1] - goal[1], target_center[0] - goal[0])
    near = np.linalg.norm(arr["position"] - goal, axis=1) < STOP_POS_TOL
    facing = np.abs(wrap(arr["yaw"] - yaw_goal)) < STOP_YAW_TOL
    hover = np.array([ph == "hover" for ph in arr["phase"]])
    ok = near & facing & hover
    # stop only on the trailing run (once stopped, stays stopped)
    stop = np.zeros(len(ok), bool)
    i = len(ok)
    while i > 0 and ok[i - 1]:
        i -= 1
    stop[i:] = True
    if not stop.any():
        stop[-1] = True
    return stop


def primitives(arr, stop, window=10, dt=0.1):
    """Primitive per frame from the next `window` frames (1 s) of realized motion."""
    pos, yaw = arr["position"], arr["yaw"]
    n = len(pos)
    out = []
    for t in range(n):
        if stop[t]:
            out.append("stop")
            continue
        e = min(n - 1, t + window)
        if e == t:
            out.append("hover")
            continue
        T = (e - t) * dt
        d = pos[e] - pos[t]
        v_h, v_z = np.linalg.norm(d[:2]) / T, d[2] / T
        w = (yaw[e] - yaw[t]) / T
        if v_h < 0.12 and abs(v_z) < 0.12:
            out.append("hover" if abs(w) < 0.15 else ("rotate left" if w > 0 else "rotate right"))
        elif abs(v_z) > max(0.15, 0.8 * v_h):
            out.append("ascend" if v_z > 0 else "descend")
        else:
            side = "left" if w > 0 else "right"
            a = abs(w)
            out.append("move forward" if a < 0.15 else f"slightly turn {side}" if a < 0.4
                       else f"turn {side}" if a < 0.8 else f"sharp turn {side}")
    return out


def segments(labels, min_len=5):
    """Run-length segments [label, start, end) with short runs merged into their predecessor."""
    segs = []
    for i, l in enumerate(labels):
        if segs and segs[-1][0] == l:
            segs[-1][2] = i + 1
        else:
            segs.append([l, i, i + 1])
    merged = []
    for s in segs:
        if merged and (s[2] - s[1] < min_len) and s[0] != "stop":
            merged[-1][2] = s[2]
        elif merged and merged[-1][0] == s[0]:
            merged[-1][2] = s[2]
        else:
            merged.append(s)
    return merged


def room_sequence(scene_meta, pos, min_dwell=8):
    """Rooms visited in order; rooms occupied for fewer than `min_dwell` frames (doorway jitter) are dropped."""
    from .tasks import room_of
    runs = []
    for p in pos:
        r = room_of(scene_meta, p)
        if r is None:
            continue
        if runs and runs[-1][0] == r:
            runs[-1][1] += 1
        else:
            runs.append([r, 1])
    seq = []
    for i, (r, n) in enumerate(runs):
        if (n >= min_dwell or i in (0, len(runs) - 1)) and (not seq or seq[-1] != r):
            seq.append(r)
    return seq


def instruction_draft(spec, room_types, rooms_seq, landmarks):
    target = spec["target"]["type"]
    t_room = spec["target"]["room_type"] or "room"
    s_room = spec["start"]["room_type"] or "room"
    if len(rooms_seq) <= 1 or rooms_seq[0] == rooms_seq[-1]:
        text = f"Find the {target} in this {t_room} and stop in front of it."
    else:
        via = []
        for r in rooms_seq[1:-1]:
            if room_types[r] not in via[-1:]:
                via.append(room_types[r])
        via_txt = f", passing through the {' and the '.join(via)}" if via else ""
        text = f"Leave the {s_room}{via_txt}, go to the {t_room}, and stop in front of the {target}."
    if landmarks:
        text += f" On the way you will see the {', the '.join(landmarks[:3])}."
    return text
