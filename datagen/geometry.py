"""Occupancy / clearance grids from ProcTHOR GLB triangles.

Grid conventions (world frame, Z-up, metres):
  index (i, j, k) <-> centre origin + (i + 0.5, j + 0.5, k + 0.5) * res
  occ[i, j, k]       True where any triangle surface passes through the voxel
  clearance[i, j, k] Euclidean distance (m) from the voxel centre to the nearest occupied voxel
  inside[i, j]       True above a floor of the house (from Floor# meshes)
"""
import json
import struct
from pathlib import Path

import numpy as np
from scipy import ndimage

from .scene_parse import GLTF_TO_WORLD, node_matrix

COMPONENT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
SKIP_PREFIX = ("Point light", "Room#exterior")


def _read_glb(path):
    data = Path(path).read_bytes()
    json_len = struct.unpack("<I", data[12:16])[0]
    gltf = json.loads(data[20:20 + json_len])
    off = 20 + json_len
    bin_len = struct.unpack("<I", data[off:off + 4])[0]
    return gltf, memoryview(data)[off + 8:off + 8 + bin_len]


def _accessor(gltf, binary, index):
    acc = gltf["accessors"][index]
    view = gltf["bufferViews"][acc["bufferView"]]
    dtype = np.dtype(COMPONENT[acc["componentType"]])
    n = NCOMP[acc["type"]]
    start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    stride = view.get("byteStride", 0) or dtype.itemsize * n
    raw = np.frombuffer(binary, dtype=np.uint8, count=stride * (acc["count"] - 1) + dtype.itemsize * n, offset=start)
    rows = np.lib.stride_tricks.as_strided(raw, shape=(acc["count"], dtype.itemsize * n), strides=(stride, 1))
    return np.ascontiguousarray(rows).view(dtype).reshape(acc["count"], n)


def load_triangles(glb_path):
    """World-frame triangles grouped by top-level node name: dict name -> (T, 3, 3) float32."""
    gltf, binary = _read_glb(glb_path)
    out = {}

    def visit(index, parent_m, top_name):
        node = gltf["nodes"][index]
        m = parent_m @ node_matrix(node)
        if "mesh" in node:
            for prim in gltf["meshes"][node["mesh"]]["primitives"]:
                if prim.get("mode", 4) != 4:
                    continue
                pos = _accessor(gltf, binary, prim["attributes"]["POSITION"]).astype(np.float64)
                idx = (_accessor(gltf, binary, prim["indices"]).reshape(-1) if "indices" in prim
                       else np.arange(len(pos)))
                world = (m[:3, :3] @ pos.T).T + m[:3, 3]
                out.setdefault(top_name, []).append(world[idx.reshape(-1, 3)].astype(np.float32))
        for c in node.get("children", []):
            visit(c, m, top_name)

    for index in gltf["scenes"][gltf.get("scene", 0)]["nodes"]:
        name = gltf["nodes"][index].get("name", "")
        if not name.startswith(SKIP_PREFIX):
            visit(index, GLTF_TO_WORLD, name)
    return {k: np.concatenate(v) for k, v in out.items()}


def sample_surface(tris, spacing):
    """Points on triangles with roughly `spacing` density (always includes vertices)."""
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    n = np.maximum(1, np.ceil(area / (spacing * spacing) * 2)).astype(np.int64)
    rep = np.repeat(np.arange(len(tris)), n)
    u, v = np.random.default_rng(0).random((2, len(rep)))
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    pts = a[rep] + u[:, None] * (b[rep] - a[rep]) + v[:, None] * (c[rep] - a[rep])
    # edges matter for thin geometry: add edge midpoints and vertices
    return np.concatenate([pts, tris.reshape(-1, 3), (a + b) / 2, (b + c) / 2, (a + c) / 2])


class Grid:
    def __init__(self, origin, shape, res):
        self.origin = np.asarray(origin, dtype=np.float64)
        self.shape = tuple(int(s) for s in shape)
        self.res = float(res)

    def to_index(self, p):
        return np.floor((np.asarray(p) - self.origin) / self.res).astype(np.int64)

    def to_world(self, ijk):
        return self.origin + (np.asarray(ijk) + 0.5) * self.res

    def valid(self, ijk):
        ijk = np.atleast_2d(ijk)
        return np.all((ijk >= 0) & (ijk < np.array(self.shape)), axis=1)


def build_maps(glb_path, res=0.1, margin=0.5):
    tris = load_triangles(glb_path)
    all_t = np.concatenate(list(tris.values()))
    lo = all_t.reshape(-1, 3).min(0) - margin
    hi = all_t.reshape(-1, 3).max(0) + margin
    grid = Grid(lo, np.ceil((hi - lo) / res).astype(int), res)

    occ = np.zeros(grid.shape, dtype=bool)
    ijk = grid.to_index(sample_surface(all_t, res * 0.4))
    ijk = ijk[grid.valid(ijk)]
    occ[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True

    inside = np.zeros(grid.shape[:2], dtype=bool)
    floor_z = np.full(grid.shape[:2], np.nan)
    for name, t in tris.items():
        if name.startswith("Floor#"):
            p = sample_surface(t, res * 0.4)
            ij = grid.to_index(p)[:, :2]
            ok = (ij >= 0).all(1) & (ij < np.array(grid.shape[:2])).all(1)
            inside[ij[ok, 0], ij[ok, 1]] = True
            floor_z[ij[ok, 0], ij[ok, 1]] = p[ok, 2]
    inside = ndimage.binary_closing(inside, iterations=2)

    clearance = ndimage.distance_transform_edt(~occ) * res
    return {"grid": grid, "occ": occ, "clearance": clearance.astype(np.float32), "inside": inside,
            "floor_z": float(np.nanmedian(floor_z)), "triangles": tris}


def flight_mask(maps, z_min=0.5, z_max=2.0, min_clearance=0.3):
    """Voxels the drone may occupy: inside the house, altitude band above the floor, enough clearance."""
    g = maps["grid"]
    z = g.origin[2] + (np.arange(g.shape[2]) + 0.5) * g.res - maps["floor_z"]
    band = (z >= z_min) & (z <= z_max)
    mask = (maps["clearance"] >= min_clearance) & maps["inside"][:, :, None] & band[None, None, :]
    # keep the largest connected component (drop pockets inside furniture etc.)
    lab, n = ndimage.label(mask, structure=np.ones((3, 3, 3)))
    if n > 1:
        sizes = ndimage.sum(mask, lab, index=np.arange(1, n + 1))
        mask = lab == (1 + int(np.argmax(sizes)))
    return mask
