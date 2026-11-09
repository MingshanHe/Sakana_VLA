"""Dynamics expert: batched L2F crazyflie + RAPTOR position controller tracking planned references.

Each drone in the batch runs one episode: (optional rotate) -> fly -> face target -> hover. With probability
`perturb_prob` a disturbance (lateral velocity kick + yaw offset) hits mid-flight; after `perturb_hold` s the expert
replans from the disturbed state with the same goal field, so the data contains deviation-and-recovery segments.

Recorded at 10 Hz from the *realized* state (never the reference): position, quaternion (w,x,y,z), linear and
angular velocity, yaw, phase, perturbation flag. Physics runs at 100 Hz.
"""
import numpy as np

from .planner import reference_trajectory, wrap

SIM_DT = 0.01
RECORD_EVERY = 10          # 100 Hz physics -> 10 Hz record
LEAD_STEPS = 55            # RAPTOR setpoint lookahead (0.55 s): mean err 0.30 m -> ~0.06 m on a 0.5 m/s circle
Z_SAG_COMP = 0.053         # RAPTOR/crazyflie hovers ~5 cm below its z setpoint; feed-forward offset
COLLISION_CLEARANCE = 0.12  # m from the nearest surface voxel centre
MAX_TRACK_ERROR = 0.6      # m from the reference => discard


def quat_from_yaw(yaw):
    return np.stack([np.cos(yaw / 2), np.zeros_like(yaw), np.zeros_like(yaw), np.sin(yaw / 2)], axis=-1)


def yaw_from_quat(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def quat_mul(a, b):
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], axis=-1)


class Episode:
    def __init__(self, spec, field, rng, perturb_prob, perturb_hold):
        self.spec, self.field = spec, field
        self.goal = np.asarray(spec["goal"])
        self.target = np.asarray(spec["target"]["center"])
        p0, yaw0 = np.asarray(spec["start"]["position"]), spec["start"]["yaw"]
        self.ref, self.phases = reference_trajectory(field, p0, yaw0, self.goal, self.target)
        self.k = 0                       # index into self.ref
        self.t = 0                       # physics step counter
        self.done = self.ref is None
        self.failed = None if self.ref is not None else "unreachable"
        self.records = []
        self.perturb_hold = perturb_hold
        self.perturb_at, self.perturb_end, self.perturbed = None, None, False
        if self.ref is not None and rng.random() < perturb_prob:
            fly = [i for i, ph in enumerate(self.phases) if ph == "fly"]
            if len(fly) > 200:
                self.perturb_at = int(rng.uniform(0.25, 0.7) * len(fly)) + fly[0]
                self.kick_dir = rng.choice([-1.0, 1.0])
                self.kick_speed = rng.uniform(0.5, 1.0)
                self.kick_yaw = rng.choice([-1.0, 1.0]) * rng.uniform(np.deg2rad(15), np.deg2rad(45))
        self.track_err = []

    def setpoint(self):
        i = min(self.k + LEAD_STEPS, len(self.ref) - 1)
        sp = self.ref[i].copy()
        sp[2] += Z_SAG_COMP
        # yaw lead is not needed (the yaw loop is fast); use the current reference yaw
        sp[3] = wrap(self.ref[min(self.k, len(self.ref) - 1), 3])
        return sp


def rollout(specs_fields, sim, maps, rng, perturb_prob=0.5, perturb_hold=0.4, max_seconds=90.0):
    """Run len(specs_fields) episodes in one batched Sim (num_drones must equal the batch). Returns episodes."""
    n = sim.num_drones
    assert len(specs_fields) <= n
    eps = [Episode(s, f, rng, perturb_prob, perturb_hold) for s, f in specs_fields]
    sim.reset()
    pos = np.zeros((n, 3), np.float32)
    yaw = np.zeros(n)
    for i, e in enumerate(eps):
        pos[i] = e.spec["start"]["position"]
        yaw[i] = e.spec["start"]["yaw"]
    for i in range(len(eps), n):
        pos[i] = [0, 0, 50.0]  # padding drones fly far away, ignored
    sim.state["position"] = pos
    sim.state["orientation"] = quat_from_yaw(yaw).astype(np.float32)
    sim.state["linear_velocity"] = np.zeros((n, 3), np.float32)
    sim.state["angular_velocity"] = np.zeros((n, 3), np.float32)

    g, clr = maps["grid"], maps["clearance"]
    actions = np.zeros((n, 4), np.float32)
    actions[:, :3] = pos
    for step in range(int(max_seconds / SIM_DT)):
        if all(e.done for e in eps):
            break
        state = {k: sim.state[k] for k in ("position", "orientation", "linear_velocity", "angular_velocity")}
        kicks = []
        for i, e in enumerate(eps):
            if e.done:
                actions[i, :3] = state["position"][i]
                actions[i, 3] = yaw_from_quat(state["orientation"][i])
                continue
            p = state["position"][i]
            # --- perturbation: kick, hold the stale setpoint, then replan from the disturbed state
            if e.perturb_at is not None and not e.perturbed and e.k >= e.perturb_at:
                e.perturbed, e.perturb_end = True, e.t + int(e.perturb_hold / SIM_DT)
                fwd = e.ref[e.k + 1, :2] - e.ref[e.k, :2]
                lat = np.array([-fwd[1], fwd[0]]) / (np.linalg.norm(fwd) + 1e-9)
                kicks.append((i, np.r_[lat * e.kick_dir * e.kick_speed, 0.0], e.kick_yaw))
            if e.perturb_end is not None and e.t == e.perturb_end:
                v0 = state["linear_velocity"][i]
                y0 = float(yaw_from_quat(state["orientation"][i]))
                ref, phases = reference_trajectory(e.field, p, y0, e.goal, e.target, v0=v0)
                if ref is None:
                    e.done, e.failed = True, "replan_failed"
                    continue
                e.ref, e.phases, e.k = ref, phases, 0
            in_hold = e.perturb_end is not None and e.t < e.perturb_end
            # --- record at 10 Hz (realized state)
            if e.t % RECORD_EVERY == 0:
                e.records.append({
                    "t": e.t * SIM_DT, "position": p.copy(), "orientation": state["orientation"][i].copy(),
                    "linear_velocity": state["linear_velocity"][i].copy(),
                    "angular_velocity": state["angular_velocity"][i].copy(),
                    "phase": "perturb" if in_hold else e.phases[min(e.k, len(e.phases) - 1)],
                    "perturbed": bool(in_hold), "ref": e.ref[min(e.k, len(e.ref) - 1)].copy(),
                })
            # --- safety / tracking checks
            ijk = g.to_index(p)
            if not g.valid(ijk)[0] or clr[tuple(ijk)] < COLLISION_CLEARANCE:
                e.done, e.failed = True, "collision"
                continue
            err = float(np.linalg.norm(p - e.ref[min(e.k, len(e.ref) - 1), :3]))
            e.track_err.append(err)
            if not in_hold and err > MAX_TRACK_ERROR:
                e.done, e.failed = True, "tracking"
                continue
            if not in_hold:
                actions[i] = e.setpoint()
                e.k += 1
            if e.k >= len(e.ref):
                e.done = True
            e.t += 1
        for i, dv, dyaw in kicks:
            v = sim.state["linear_velocity"]
            v[i] += dv.astype(np.float32)
            sim.state["linear_velocity"] = v
            q = sim.state["orientation"]
            q[i] = quat_mul(quat_from_yaw(np.array(dyaw)), q[i])
            sim.state["orientation"] = q
        sim.step(actions)
    for e in eps:
        if not e.done:
            e.done, e.failed = True, "timeout"
    return eps
