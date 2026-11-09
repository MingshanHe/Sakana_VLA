"""Parse a ProcTHOR GLB (as cached by hyperlux/conta) into scene metadata.

Output per house: objects (instance roots carrying rl_tools.segmentation_class) with world AABBs,
rooms with floor AABB/polygon, and semantic-class counts. Coordinates are converted from glTF
(Y-up) to the hyperlux world frame (Z-up) by `GLTF_TO_WORLD` (verified in spike_render.py).

    python -m datagen.scene_parse ProcTHOR-Train-1 [--out data/scenes]
"""
import argparse
import json
import re
import struct
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# glTF Y-up -> world Z-up: (x, y, z)_gltf -> (x, -z, y)_world. Checked against renderer.scene_bounds.
GLTF_TO_WORLD = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)

# Structural / clutter types never used as navigation targets.
NON_TARGET = {"wall", "wall inside", "floor", "ceiling", "room", "doorway", "doorframe", "door", "window", "mesh",
              "point light", "extra objects", "light switch", "painting", "poster", "curtains", "ceiling room",
              "floor room", "doorway open", "doorframe double", "window hung", "wall decor photo", ""}
# Small/flat objects are hard to see from ~1 m at 256 px: need sorted AABB extents e1 >= 0.3, e2 >= 0.4.
MIN_TARGET_MID, MIN_TARGET_MAX = 0.3, 0.4
LABELS_PATH = Path(__file__).with_name("class_labels.json")  # segmentation_class -> label, from build_labels()

# Room type inference from contents (ProcTHOR room types: Kitchen, LivingRoom, Bedroom, Bathroom).
ROOM_CUES = {
    "bathroom": {"toilet", "plunger", "bathroom faucet", "soap bar", "toilet paper used up"},
    "kitchen": {"fridge", "microwave", "coffee machine", "toaster", "kettle", "pot", "pan"},
    "bedroom": {"bed", "alarm clock"},
    "living room": {"sofa", "television", "tv stand", "coffee table", "armchair"},
}

def read_glb(path):
    data = Path(path).read_bytes()
    if data[:4] != b"glTF":
        raise ValueError(f"{path}: not a GLB")
    json_len = struct.unpack("<I", data[12:16])[0]
    return json.loads(data[20:20 + json_len])


def node_matrix(node):
    if "matrix" in node:
        return np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T  # glTF is column-major
    m = np.eye(4)
    if "scale" in node:
        m = np.diag([*node["scale"], 1.0]) @ m
    if "rotation" in node:
        x, y, z, w = node["rotation"]
        r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        rm = np.eye(4)
        rm[:3, :3] = r
        m = rm @ m
    if "translation" in node:
        t = np.eye(4)
        t[:3, 3] = node["translation"]
        m = t @ m
    return m


def mesh_local_aabb(gltf, mesh_index):
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for prim in gltf["meshes"][mesh_index]["primitives"]:
        acc = gltf["accessors"][prim["attributes"]["POSITION"]]
        lo = np.minimum(lo, acc["min"])
        hi = np.maximum(hi, acc["max"])
    return lo, hi


def transform_aabb(m, lo, hi):
    corners = np.array([[x, y, z, 1.0] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    w = (m @ corners.T).T[:, :3]
    return w.min(0), w.max(0)


def subtree_aabb(gltf, index, parent_m, cache):
    node = gltf["nodes"][index]
    m = parent_m @ node_matrix(node)
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    if "mesh" in node:
        mlo, mhi = cache.setdefault(node["mesh"], mesh_local_aabb(gltf, node["mesh"]))
        wlo, whi = transform_aabb(m, mlo, mhi)
        lo, hi = np.minimum(lo, wlo), np.maximum(hi, whi)
    for c in node.get("children", []):
        clo, chi = subtree_aabb(gltf, c, m, cache)
        lo, hi = np.minimum(lo, clo), np.maximum(hi, chi)
    return lo, hi


def clean_name(name):
    """'Laundry_Hamper_1_4' -> 'laundry hamper', 'Countertop_L_4x4' -> 'countertop', 'RoboTHOR_dresser_rast_v' -> 'dresser rast'."""
    tokens = [t for t in re.split(r"[_|#\s]+", name) if t]
    if tokens and tokens[0] == "RoboTHOR":
        tokens = tokens[1:]
    keep = [t for t in tokens if not re.fullmatch(r"\d+|\d+x\d+|[A-Za-z]|v\d*", t)]
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", " ".join(keep)).lower().strip()


def object_type(name, seg):
    labels = _labels()
    return labels.get(str(seg)) or clean_name(name)


_LABELS = None


def _labels():
    global _LABELS
    if _LABELS is None:
        _LABELS = json.loads(LABELS_PATH.read_text()) if LABELS_PATH.exists() else {}
    return _LABELS


def parse(glb_path, house_name):
    gltf = read_glb(glb_path)
    cache = {}
    root = GLTF_TO_WORLD
    objects, rooms = [], {}
    for index in gltf["scenes"][gltf.get("scene", 0)]["nodes"]:
        node = gltf["nodes"][index]
        name, extras = node.get("name", ""), node.get("extras") or {}
        lo, hi = subtree_aabb(gltf, index, root, cache)
        if not np.all(np.isfinite(lo)):
            continue
        if name.startswith("Floor#room|"):
            room_id = name.split("#", 1)[1][:-2]  # "room|2|f" -> "room|2"
            rooms[room_id] = {"id": room_id, "aabb": [lo.tolist(), hi.tolist()]}
            continue
        seg = (extras.get("rl_tools") or {}).get("segmentation_class")
        if seg is None:
            continue
        objects.append({"name": name, "type": object_type(name, seg), "segmentation_class": int(seg),
                        "aabb": [lo.tolist(), hi.tolist()], "center": ((lo + hi) / 2).tolist(),
                        "extent": (hi - lo).tolist()})

    # room assignment by floor AABB containment of the object's XY centre
    for o in objects:
        cx, cy = o["center"][:2]
        o["room"] = next((r["id"] for r in rooms.values()
                          if r["aabb"][0][0] <= cx <= r["aabb"][1][0] and r["aabb"][0][1] <= cy <= r["aabb"][1][1]), None)
    contents = defaultdict(set)
    for o in objects:
        if o["room"]:
            contents[o["room"]].add(o["type"])
    for rid, r in rooms.items():
        scores = {t: len(cues & contents[rid]) for t, cues in ROOM_CUES.items()}
        best = max(scores, key=scores.get)
        r["type"] = best if scores[best] > 0 else "room"
        r["contents"] = sorted(contents[rid])

    # target eligibility: semantic class unique within the house (semantic mask == instance mask), big enough
    class_count = Counter(o["segmentation_class"] for o in objects)
    for i, o in enumerate(objects):
        o["id"] = i
        o["unique_class"] = class_count[o["segmentation_class"]] == 1
        e = sorted(o["extent"])
        o["target_ok"] = (o["unique_class"] and o["type"] not in NON_TARGET and not o["type"].startswith("wall") and o["room"] is not None
                          and e[1] >= MIN_TARGET_MID and e[2] >= MIN_TARGET_MAX)

    all_lo = np.min([o["aabb"][0] for o in objects] + [r["aabb"][0] for r in rooms.values()], axis=0)
    all_hi = np.max([o["aabb"][1] for o in objects] + [r["aabb"][1] for r in rooms.values()], axis=0)
    return {"house": house_name, "glb": str(glb_path), "bounds": [all_lo.tolist(), all_hi.tolist()],
            "rooms": list(rooms.values()), "objects": objects,
            "num_targets": sum(o["target_ok"] for o in objects)}


def resolve_glb(house_name):
    """Path of the cached GLB for e.g. 'ProcTHOR-Train-1' (downloads through hyperlux if missing)."""
    cache = Path.home() / ".cache" / "conta"
    for index in cache.glob("index-*.json"):
        for e in json.loads(index.read_text())["entries"]:
            if e["description"] == f"{house_name}.glb" and (cache / e["sha1"]).exists():
                return str(cache / e["sha1"])
    import hyperlux  # needs a GPU node (CUDA target detection) for the native core
    ref = hyperlux.datasets.procthor.Train200()[house_name].reference
    return str(ref)


def build_labels(glbs):
    """Majority-vote a readable label per segmentation_class over many houses (non-RoboTHOR names preferred)."""
    votes = defaultdict(Counter)
    for g in glbs:
        gltf = read_glb(g)
        for index in gltf["scenes"][0]["nodes"]:
            node = gltf["nodes"][index]
            seg = ((node.get("extras") or {}).get("rl_tools") or {}).get("segmentation_class")
            if seg is None:
                continue
            name = node.get("name", "")
            votes[int(seg)][clean_name(name)] += 1 if name.startswith("RoboTHOR") else 10
    labels = {str(k): v.most_common(1)[0][0] for k, v in sorted(votes.items())}
    LABELS_PATH.write_text(json.dumps(labels, indent=1))
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("houses", nargs="+")
    ap.add_argument("--out", default="data/scenes")
    ap.add_argument("--build-labels", action="store_true", help="rebuild class_labels.json from these houses first")
    args = ap.parse_args()
    if args.build_labels:
        labels = build_labels([resolve_glb(h) for h in args.houses])
        print(f"{len(labels)} class labels -> {LABELS_PATH}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for h in args.houses:
        meta = parse(resolve_glb(h), h)
        (out / f"{h}.json").write_text(json.dumps(meta, indent=1))
        rooms = ", ".join(f"{r['id']}:{r['type']}" for r in meta["rooms"])
        targets = sorted({o["type"] for o in meta["objects"] if o["target_ok"]})
        print(f"{h}: {len(meta['objects'])} objects, {meta['num_targets']} targets {targets}; rooms {rooms}")


if __name__ == "__main__":
    main()
