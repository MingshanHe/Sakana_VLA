"""Episode sampling: target object -> goal viewpoint (visible, safe) -> start (geodesic 3-12 m away)."""
import numpy as np

from .planner import GoalField


def line_of_sight(maps, eye, target_aabb, step=0.05, skip=0.15):
    """Ray from eye towards the AABB centre is free of occupied voxels until it enters the (padded) AABB."""
    lo, hi = np.asarray(target_aabb[0]) - 0.1, np.asarray(target_aabb[1]) + 0.1
    c = (lo + hi) / 2
    d = c - eye
    n = int(np.linalg.norm(d) / step)
    g = maps["grid"]
    for k in range(int(skip / step), n):
        p = eye + d * (k / n)
        if np.all(p >= lo) and np.all(p <= hi):
            return True
        ijk = g.to_index(p)
        if not g.valid(ijk)[0] or maps["occ"][tuple(ijk)]:
            return False
    return True


def in_fov(eye, yaw, point, hfov=np.deg2rad(90), vfov=np.deg2rad(90), margin=0.8):
    d = np.asarray(point) - eye
    az = np.arctan2(d[1], d[0]) - yaw
    az = (az + np.pi) % (2 * np.pi) - np.pi
    el = np.arctan2(d[2], np.linalg.norm(d[:2]))
    return abs(az) < margin * hfov / 2 and abs(el) < margin * vfov / 2


def sample_goal_viewpoint(maps, mask, obj, rng, tries=200, dist=(0.6, 1.4)):
    """Hover point near the object: horizontal standoff from the AABB, altitude near the object's top."""
    lo, hi = np.asarray(obj["aabb"][0]), np.asarray(obj["aabb"][1])
    c = (lo + hi) / 2
    radius = 0.5 * np.linalg.norm((hi - lo)[:2])
    floor = maps["floor_z"]
    g = maps["grid"]
    for _ in range(tries):
        ang = rng.uniform(0, 2 * np.pi)
        r = radius + rng.uniform(*dist)
        z = np.clip(hi[2] + rng.uniform(0.1, 0.6), floor + 0.6, floor + 1.8)
        p = np.array([c[0] + r * np.cos(ang), c[1] + r * np.sin(ang), z])
        ijk = g.to_index(p)
        if not g.valid(ijk)[0] or not mask[tuple(ijk)]:
            continue
        yaw = np.arctan2(c[1] - p[1], c[0] - p[0])
        if in_fov(p, yaw, c) and line_of_sight(maps, p, obj["aabb"]):
            return g.to_world(ijk), c
    return None, None


def sample_starts(field, mask, rng, n, geo_range=(3.0, 12.0), tries=2000):
    cells = field.cells
    reach = np.isfinite(field.dist)
    ok = reach & (field.dist >= geo_range[0]) & (field.dist <= geo_range[1])
    idx = np.flatnonzero(ok)
    if len(idx) == 0:
        return []
    pick = rng.choice(idx, size=min(n, len(idx)), replace=False)
    out = []
    for i in pick:
        p = field.maps["grid"].to_world(cells[i]) + rng.uniform(-0.04, 0.04, 3)
        out.append({"position": p, "yaw": float(rng.uniform(-np.pi, np.pi)), "geodesic": float(field.dist[i])})
    return out


def sample_episodes(scene_meta, maps, mask, rng, episodes_per_target=5, max_targets=None):
    """Yields episode specs (start, goal viewpoint, target) and the GoalField used to plan/replan them."""
    targets = [o for o in scene_meta["objects"] if o["target_ok"]]
    rng.shuffle(targets)
    if max_targets:
        targets = targets[:max_targets]
    rooms = {r["id"]: r for r in scene_meta["rooms"]}
    for obj in targets:
        goal, target_c = sample_goal_viewpoint(maps, mask, obj, rng)
        if goal is None:
            continue
        field = GoalField(maps, mask, maps["grid"].to_index(goal))
        for s in sample_starts(field, mask, rng, episodes_per_target):
            start_room = room_of(scene_meta, s["position"])
            yield {
                "house": scene_meta["house"],
                "target": {"id": obj["id"], "name": obj["name"], "type": obj["type"],
                           "segmentation_class": obj["segmentation_class"], "aabb": obj["aabb"],
                           "center": target_c.tolist(), "room": obj["room"],
                           "room_type": rooms[obj["room"]]["type"] if obj["room"] in rooms else None},
                "goal": goal.tolist(),
                "start": {"position": s["position"].tolist(), "yaw": s["yaw"], "room": start_room,
                          "room_type": rooms[start_room]["type"] if start_room in rooms else None},
                "geodesic": s["geodesic"],
            }, field


def room_of(scene_meta, p):
    for r in scene_meta["rooms"]:
        lo, hi = r["aabb"]
        if lo[0] <= p[0] <= hi[0] and lo[1] <= p[1] <= hi[1]:
            return r["id"]
    return None
