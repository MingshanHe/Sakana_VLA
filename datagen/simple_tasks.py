"""Simple single-room tasks: the target is in the same room with a free straight line from the start.

Two tasks, no room changes and no search beyond one in-place scan:
  fly_to  fly to a viewpoint in front of the target, face it, hover (stop).
  land    fly above a flat spot on the target (tables, desks, beds, ...), descend to ~0.2 m above it, hover (stop).
Two start modes:
  front   the target is in view at the start (yaw within +-30 deg of its bearing).
  scan    the target is out of view; the expert always turns LEFT (counter-clockwise) until it faces the path,
          so "target not visible -> turn left" is a single-valued rule a memoryless policy can learn.
"""
import numpy as np

from .planner import GoalField, segment_clear, wrap
from .tasks import in_fov, line_of_sight, room_of, sample_goal_viewpoint

LANDABLE = ("table", "desk", "countertop", "bed", "dresser", "tv stand")
LAND_MIN_FOOTPRINT, LAND_MAX_TOP = 0.5, 1.3     # m (both horizontal extents; top above the floor)
LAND_VOXELS_ABOVE, PRELAND_VOXELS_ABOVE = 2, 5  # voxel centres above the top surface voxel: ~0.2 m and ~0.5 m
TEMPLATES = {
    "fly_to": ["Fly to the {t}.", "Go to the {t} and stop in front of it.", "Find the {t} and fly to it."],
    "land": ["Land on the {t}.", "Fly to the {t} and land on it.", "Find the {t} and land on it."],
}


def landable(obj, floor_z):
    e = obj["extent"]
    return (any(k in obj["type"] for k in LANDABLE) and min(e[0], e[1]) >= LAND_MIN_FOOTPRINT
            and obj["aabb"][1][2] - floor_z <= LAND_MAX_TOP)


def sample_land_spot(maps, mask, obj, rng, tries=30, margin=0.2):
    """A flat free spot on top of the object. Returns (pre-land point in the flight mask, land point) or (None, None)."""
    g, occ, clr = maps["grid"], maps["occ"], maps["clearance"]
    lo, hi = np.asarray(obj["aabb"][0]), np.asarray(obj["aabb"][1])
    c = (lo + hi) / 2
    half = np.maximum((hi - lo)[:2] / 2 - margin, 0.0)
    for n in range(tries):
        xy = c[:2] if n == 0 else c[:2] + rng.uniform(-half, half)
        i, j, k_lo = g.to_index([xy[0], xy[1], lo[2]])
        k_hi = min(g.shape[2] - 1, g.to_index([xy[0], xy[1], hi[2]])[2] + 1)
        if not (0 <= i < g.shape[0] and 0 <= j < g.shape[1]):
            continue
        col = np.flatnonzero(occ[i, j, max(k_lo, 0):k_hi + 1])
        if len(col) == 0:
            continue
        k_top = max(k_lo, 0) + int(col[-1])
        k_land, k_pre = k_top + LAND_VOXELS_ABOVE, k_top + PRELAND_VOXELS_ABOVE
        if k_pre >= g.shape[2] or not mask[i, j, k_pre]:
            continue
        # flat and free: nothing but the surface below is nearer than the surface itself, all the way up
        if clr[i, j, k_land] < LAND_VOXELS_ABOVE * g.res - 0.01 or occ[i, j, k_top + 1:k_pre + 1].any():
            continue
        return g.to_world([i, j, k_pre]), g.to_world([i, j, k_land])
    return None, None


def sample_simple_starts(scene_meta, maps, field, obj, goal, target_c, task, rng, n, dist=(1.5, 5.0), max_tests=400,
                         front_prob=0.6):
    g = maps["grid"]
    ok = np.isfinite(field.dist) & (field.dist >= dist[0])
    out = []
    for i in rng.permutation(np.flatnonzero(ok))[:max_tests]:
        if len(out) >= n:
            break
        p = g.to_world(field.cells[i])
        d_h = np.linalg.norm((goal - p)[:2])
        if not (dist[0] <= d_h <= dist[1]) or room_of(scene_meta, p) != obj["room"]:
            continue
        if not segment_clear(maps, p, goal, 0.25) or not line_of_sight(maps, p, obj["aabb"]):
            continue
        bearing = float(np.arctan2(target_c[1] - p[1], target_c[0] - p[0]))
        path_dir = float(np.arctan2(goal[1] - p[1], goal[0] - p[0]))
        if abs(wrap(path_dir - bearing)) > np.deg2rad(30):  # fly towards what you look at (large objects up close)
            continue
        if task == "fly_to":  # approach from the side the viewpoint is on (no detour around the object)
            view_dir = float(np.arctan2(target_c[1] - goal[1], target_c[0] - goal[0]))
            if abs(wrap(view_dir - bearing)) > np.deg2rad(45):
                continue
        if rng.random() < front_prob:
            mode, yaw = "front", bearing + rng.uniform(-np.deg2rad(30), np.deg2rad(30))
        else:
            mode = "scan"
            for _ in range(20):
                yaw = path_dir - rng.uniform(np.deg2rad(65), np.deg2rad(290))  # needs a 65..290 deg left turn
                if not in_fov(p, yaw, target_c, margin=1.2):
                    break
            else:
                continue
        out.append({"position": p + rng.uniform(-0.04, 0.04, 3), "yaw": float(wrap(yaw)), "geodesic": float(field.dist[i]), "mode": mode,
                    "distance": float(np.linalg.norm(goal - p)), "yaw_goal": bearing})
    return out


def sample_simple_episodes(scene_meta, maps, mask, rng, episodes_per_target=2, tasks=("fly_to", "land")):
    """Yields (spec, GoalField). spec carries the extra keys task / start_mode / stop_point / plan / instruction."""
    targets = [o for o in scene_meta["objects"] if o["target_ok"]]
    rng.shuffle(targets)
    rooms = {r["id"]: r for r in scene_meta["rooms"]}
    g = maps["grid"]
    for obj in targets:
        for task in tasks:
            if task == "land":
                if not landable(obj, maps["floor_z"]):
                    continue
                goal, stop_point = sample_land_spot(maps, mask, obj, rng)
                target_c = (np.asarray(obj["aabb"][0]) + np.asarray(obj["aabb"][1])) / 2
            else:
                goal, target_c = sample_goal_viewpoint(maps, mask, obj, rng)
                stop_point = goal
            if goal is None:
                continue
            field = GoalField(maps, mask, g.to_index(goal))
            for s in sample_simple_starts(scene_meta, maps, field, obj, goal, target_c, task, rng, episodes_per_target):
                plan = {"first_turn": 1.0 if s["mode"] == "scan" else None}
                if task == "land":
                    plan.update({"yaw_goal": s["yaw_goal"], "land_z": float(stop_point[2])})
                room_type = rooms[obj["room"]]["type"] if obj["room"] in rooms else None
                yield {
                    "house": scene_meta["house"], "task": task, "start_mode": s["mode"],
                    "target": {"id": obj["id"], "name": obj["name"], "type": obj["type"],
                               "segmentation_class": obj["segmentation_class"], "aabb": obj["aabb"],
                               "center": np.asarray(target_c).tolist(), "room": obj["room"], "room_type": room_type},
                    "goal": np.asarray(goal).tolist(), "stop_point": np.asarray(stop_point).tolist(),
                    "start": {"position": s["position"].tolist(), "yaw": s["yaw"], "room": obj["room"],
                              "room_type": room_type},
                    "geodesic": s["geodesic"], "distance": s["distance"], "plan": plan,
                    "instruction": str(rng.choice(TEMPLATES[task])).format(t=obj["type"]),
                }, field
