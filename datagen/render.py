"""Render recorded drone poses with hyperlux: RGB + depth + semantic segmentation -> target bbox.

The camera is rigidly mounted on the body (forward-looking, `MOUNT_OFFSET` ahead of the CoM, pitch
`MOUNT_PITCH_DEG` down), so frames include the real roll/pitch of the dynamics.
"""
import numpy as np

WIDTH = HEIGHT = 256
FOV_DEG = 90.0
MOUNT_OFFSET = np.array([0.03, 0.0, 0.0])
MOUNT_PITCH_DEG = 0.0
DEPTH_MAX = 10.0  # m, clipped; stored as uint16 millimetres


def quat_to_rot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def camera_pose(position, quat):
    """World eye, forward, up for the body-mounted camera (FLU body frame)."""
    r = quat_to_rot(quat)
    pitch = np.deg2rad(MOUNT_PITCH_DEG)
    fwd_body = np.array([np.cos(pitch), 0.0, -np.sin(pitch)])
    up_body = np.array([np.sin(pitch), 0.0, np.cos(pitch)])
    return position + r @ MOUNT_OFFSET, r @ fwd_body, r @ up_body


class Renderer:
    def __init__(self, glb_path, batch=64, width=WIDTH, height=HEIGHT, fidelity="high"):
        import hyperlux
        from hyperlux.camera import CameraSpec, Depth, RGB, Segmentation
        self.RGB, self.Depth, self.Seg = RGB("rgba8"), Depth(), Segmentation("uint32", labels="semantic")
        self.scene = hyperlux.load_scene(glb_path, preserve_instances=True)
        self.r = hyperlux.HyperLux(width=width, height=height, batch_size=batch,
                                   cameras=(CameraSpec("front", (self.RGB, self.Depth, self.Seg)),), fidelity=fidelity)
        self.r.init(self.scene)
        self.batch = batch

    def render(self, positions, quats):
        """positions (N,3), quats (N,4 wxyz) -> rgb uint8 (N,H,W,3), depth float32 (N,H,W), seg uint32 (N,H,W)."""
        n = len(positions)
        rgb, depth, seg = [], [], []
        for s in range(0, n, self.batch):
            cams = []
            for p, q in zip(positions[s:s + self.batch], quats[s:s + self.batch]):
                eye, fwd, up = camera_pose(np.asarray(p, float), np.asarray(q, float))
                cams.append(self.r.camera(position=tuple(eye), look_at=tuple(eye + fwd), up=tuple(up), fov=FOV_DEG))
            m = len(cams)
            while len(cams) < self.batch:
                cams.append(cams[-1])
            self.r.set_cameras(np.stack(cams).astype(np.float32))
            self.r.render()
            rgb.append(self.r.outputs[self.RGB].read()[:m, ..., :3].copy())
            depth.append(self.r.outputs[self.Depth].read()[:m].copy())
            seg.append(self.r.outputs[self.Seg].read()[:m].copy())
        rgb, depth, seg = np.concatenate(rgb), np.concatenate(depth), np.concatenate(seg)
        return rgb, depth.reshape(depth.shape[:3]), seg.reshape(seg.shape[:3])


def target_bbox(seg, cls, min_pixels=20):
    """Normalized [x, y, w, h] (top-left origin, like AeroBrain) of the target's semantic mask; zeros if absent."""
    out = np.zeros((len(seg), 4), np.float32)
    pix = np.zeros(len(seg), np.int64)
    h, w = seg.shape[1:]
    for i, s in enumerate(seg):
        ys, xs = np.nonzero(s == cls)
        pix[i] = len(xs)
        if len(xs) >= min_pixels:
            out[i] = [xs.min() / w, ys.min() / h, (xs.max() + 1 - xs.min()) / w, (ys.max() + 1 - ys.min()) / h]
    return out, pix


def depth_to_mm(depth):
    d = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    return (np.clip(d, 0, DEPTH_MAX) * 1000).astype(np.uint16)
