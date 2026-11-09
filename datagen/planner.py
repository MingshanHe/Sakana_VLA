"""Geometric planning on the flight mask: Dijkstra field from the goal, shortcut, spline, time/yaw profile.

One Dijkstra per goal viewpoint serves every start (and every mid-flight replan) for that goal.
"""
import numpy as np
from scipy import interpolate, sparse
from scipy.sparse import csgraph

OFFSETS = np.array([(di, dj, dk) for di in (-1, 0, 1) for dj in (-1, 0, 1) for dk in (-1, 0, 1)
                    if (di, dj, dk) != (0, 0, 0)])


class GoalField:
    """Shortest-path tree to one goal voxel over the flight mask (edge cost: length x clearance penalty)."""

    def __init__(self, maps, mask, goal_ijk, clearance_soft=0.6, penalty=2.0):
        self.maps, self.mask = maps, mask
        g = maps["grid"]
        self.cells = np.argwhere(mask)
        self.lookup = -np.ones(g.shape, dtype=np.int64)
        self.lookup[tuple(self.cells.T)] = np.arange(len(self.cells))
        clr = maps["clearance"][tuple(self.cells.T)]
        node_cost = 1.0 + penalty * np.clip(clearance_soft - clr, 0, None) / clearance_soft
        rows, cols, w = [], [], []
        shape = np.array(g.shape)
        for off in OFFSETS:
            nb = self.cells + off
            ok = np.all((nb >= 0) & (nb < shape), axis=1)
            idx = np.full(len(nb), -1)
            idx[ok] = self.lookup[tuple(nb[ok].T)]
            ok = idx >= 0
            src, dst = np.flatnonzero(ok), idx[ok]
            rows.append(src)
            cols.append(dst)
            w.append(np.linalg.norm(off) * g.res * 0.5 * (node_cost[src] + node_cost[dst]))
        graph = sparse.csr_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))),
                                  shape=(len(self.cells),) * 2)
        self.goal = int(self.lookup[tuple(goal_ijk)])
        if self.goal < 0:
            raise ValueError("goal voxel is not in the flight mask")
        self.dist, self.pred = csgraph.dijkstra(graph, indices=self.goal, return_predecessors=True)

    def nearest_node(self, p):
        """Flight-mask node nearest to world point p (snaps perturbed / slightly unsafe states)."""
        g = self.maps["grid"]
        ijk = g.to_index(p)
        if g.valid(ijk)[0] and self.lookup[tuple(ijk)] >= 0:
            return int(self.lookup[tuple(ijk)])
        d = np.linalg.norm(self.cells - ijk, axis=1)
        return int(np.argmin(d))

    def geodesic(self, p):
        return float(self.dist[self.nearest_node(p)])

    def path(self, p):
        """World-frame voxel path from p to the goal (empty if unreachable)."""
        n = self.nearest_node(p)
        if not np.isfinite(self.dist[n]):
            return np.zeros((0, 3))
        nodes = [n]
        while nodes[-1] != self.goal:
            nodes.append(int(self.pred[nodes[-1]]))
        return self.maps["grid"].to_world(self.cells[nodes])


def segment_clear(maps, a, b, min_clearance, step=0.05):
    n = max(2, int(np.ceil(np.linalg.norm(b - a) / step)) + 1)
    pts = a + np.linspace(0, 1, n)[:, None] * (b - a)
    g = maps["grid"]
    ijk = g.to_index(pts)
    if not g.valid(ijk).all():
        return False
    return bool(np.all(maps["clearance"][tuple(ijk.T)] >= min_clearance))


def shortcut(maps, pts, min_clearance):
    out, i = [pts[0]], 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not segment_clear(maps, pts[i], pts[j], min_clearance):
            j -= 1
        out.append(pts[j])
        i = j
    return np.array(out)


def densify(pts, max_seg=0.6):
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / max_seg)))
        out.extend(a + (b - a) * (k / n) for k in range(1, n + 1))
    return np.array(out)


def smooth_path(maps, raw, min_clearance=0.25, start=None):
    """Shortcut + chord-length cubic spline. Returns a callable s -> xyz over arc length and its length."""
    pts = raw if start is None else np.vstack([start, raw])
    pts = shortcut(maps, pts, min_clearance)
    for attempt in range(4):
        dense = densify(pts, max_seg=0.6 / (attempt + 1))
        d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))]
        keep = np.r_[True, np.diff(d) > 1e-6]
        dense, d = dense[keep], d[keep]
        if len(dense) < 2:
            return None, 0.0
        if len(dense) == 2:
            spline = interpolate.interp1d(d, dense, axis=0)
        else:
            spline = interpolate.make_interp_spline(d, dense, k=min(3, len(dense) - 1), bc_type=None)
        s = np.linspace(0, d[-1], max(2, int(d[-1] / 0.03)))
        curve = spline(s)
        ijk = maps["grid"].to_index(curve)
        if maps["grid"].valid(ijk).all() and np.all(maps["clearance"][tuple(ijk.T)] >= min_clearance * 0.8):
            return spline, float(d[-1])
        pts = raw if start is None else np.vstack([start, raw])  # fall back to the voxel path, denser
    return interpolate.interp1d(d, dense, axis=0), float(d[-1])


def speed_profile(length, v_max=0.5, a_max=0.5, v0=0.0, dt=0.01):
    """Trapezoidal arc-length profile s(t) from speed v0 to rest at `length`."""
    s, v, out = 0.0, v0, [0.0]
    while s < length - 1e-4:
        v_brake = np.sqrt(max(0.0, 2 * a_max * (length - s)))
        v = min(v + a_max * dt, v_max, v_brake)
        v = max(v, 0.02)
        s = min(length, s + v * dt)
        out.append(s)
    return np.array(out)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def yaw_profile(xyz, yaw0, yaw_goal, dt=0.01, yaw_rate=1.2, face_dist=1.0):
    """Yaw follows the travel direction, blends to `yaw_goal` over the last `face_dist` metres, rate-limited."""
    vel = np.gradient(xyz, dt, axis=0)
    heading = np.arctan2(vel[:, 1], vel[:, 0])
    speed = np.linalg.norm(vel[:, :2], axis=1)
    remaining = np.r_[np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))[::-1], 0.0]
    remaining = np.r_[remaining[1:], 0.0] if len(remaining) > len(xyz) else remaining
    w = np.clip(1 - remaining / face_dist, 0, 1)
    desired = np.where(speed > 0.05, heading, np.nan)
    # hold the last valid heading where hovering
    last = yaw0
    for i in range(len(desired)):
        if np.isnan(desired[i]):
            desired[i] = last
        last = desired[i]
    desired = desired + w * wrap(yaw_goal - desired)
    out = np.empty(len(xyz))
    y = yaw0
    for i in range(len(xyz)):
        y = y + np.clip(wrap(desired[i] - y), -yaw_rate * dt, yaw_rate * dt)
        out[i] = y
    return out


def rotate_in_place(p, yaw0, yaw1, dt=0.01, yaw_rate=0.8):
    n = max(1, int(np.ceil(abs(wrap(yaw1 - yaw0)) / (yaw_rate * dt))))
    yaw = yaw0 + wrap(yaw1 - yaw0) * np.linspace(0, 1, n + 1)[1:]
    return np.repeat(p[None], n, axis=0), yaw


def reference_trajectory(field, p0, yaw0, goal_xyz, target_xyz, dt=0.01, v_max=0.5, a_max=0.5, v0=None,
                         rotate_first_threshold=np.deg2rad(60), final_hover=1.0):
    """Full reference (N, 4) = [x, y, z, yaw] at dt from p0/yaw0 to the goal viewpoint, then hover facing target.

    v0: current velocity (replans start moving). Returns (ref, phases) with phases[i] in
    {"rotate", "fly", "hover"}; None if unreachable.
    """
    raw = field.path(p0)
    if len(raw) == 0:
        return None, None
    raw = np.vstack([raw[:-1], goal_xyz]) if len(raw) > 1 else np.vstack([p0, goal_xyz])
    spline, length = smooth_path(field.maps, raw, start=np.asarray(p0, dtype=np.float64))
    yaw_goal = float(np.arctan2(target_xyz[1] - goal_xyz[1], target_xyz[0] - goal_xyz[0]))
    parts, phases = [], []
    speed0 = 0.0 if v0 is None else float(np.linalg.norm(v0))
    if spline is not None and length > 0.05:
        s = speed_profile(length, v_max, a_max, v0=min(speed0, v_max), dt=dt)
        xyz = spline(s)
        start_dir = np.arctan2(*(xyz[min(len(xyz) - 1, 30)] - xyz[0])[[1, 0]])
        if speed0 < 0.1 and abs(wrap(start_dir - yaw0)) > rotate_first_threshold:
            rp, ry = rotate_in_place(np.asarray(p0, dtype=np.float64), yaw0, start_dir, dt)
            parts.append(np.c_[rp, ry])
            phases += ["rotate"] * len(rp)
            yaw0 = start_dir
        yaw = yaw_profile(xyz, yaw0, yaw_goal, dt)
        parts.append(np.c_[xyz, yaw])
        phases += ["fly"] * len(xyz)
        end_p, end_yaw = xyz[-1], yaw[-1]
    else:
        end_p, end_yaw = np.asarray(p0, dtype=np.float64), yaw0
    if abs(wrap(yaw_goal - end_yaw)) > np.deg2rad(5):
        rp, ry = rotate_in_place(end_p, end_yaw, yaw_goal, dt)
        parts.append(np.c_[rp, ry])
        phases += ["rotate"] * len(rp)
    n_hover = int(final_hover / dt)
    parts.append(np.c_[np.repeat(end_p[None], n_hover, 0), np.full(n_hover, yaw_goal)])
    phases += ["hover"] * n_hover
    ref = np.concatenate(parts)
    ref[:, 3] = np.unwrap(ref[:, 3])
    return ref, phases
