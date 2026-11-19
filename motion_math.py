"""Rotations, forward kinematics, loops and resampling for motion clips (numpy, float64).

Conventions: +Y up, the body faces +Z, the person's own left is +X. Quaternions are (x, y, z, w).
A clip is a list of local joint rotations per frame (the root's is its orientation in the world)
and the root joint's position in the world per frame, in metres.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def mat_to_quat(m):
    m = np.asarray(m, dtype=np.float64)
    return Rotation.from_matrix(m.reshape(-1, 3, 3)).as_quat().reshape(m.shape[:-2] + (4,))


def quat_to_mat(q):
    q = np.asarray(q, dtype=np.float64)
    return Rotation.from_quat(q.reshape(-1, 4)).as_matrix().reshape(q.shape[:-1] + (3, 3))


def continuous(q):
    """Flip signs so each frame's quaternion is on the same side as the previous frame's: q and -q
    are the same rotation, but interpolating between opposite signs swings the long way round."""
    q = np.array(q, dtype=np.float64)
    for f in range(1, len(q)):
        flip = np.sum(q[f] * q[f - 1], axis=-1) < 0
        q[f][flip] *= -1
    return q


def slerp(a, b, t):
    """Spherical interpolation from a to b (..., 4) by t (broadcast over the leading axes)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    d = np.sum(a * b, axis=-1)
    b = np.where((d < 0)[..., None], -b, b)
    d = np.abs(d)
    t = np.broadcast_to(t, d.shape)[..., None]
    theta = np.arccos(np.clip(d, -1.0, 1.0))[..., None]
    s = np.sin(theta)
    near = s < 1e-6
    safe = np.where(near, 1.0, s)
    wa = np.where(near, 1.0 - t, np.sin((1.0 - t) * theta) / safe)
    wb = np.where(near, t, np.sin(t * theta) / safe)
    out = wa * a + wb * b
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


def rot_y(angle):
    """Rotation by angle (radians) about +Y; a positive angle turns +Z towards +X."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def heading(root_rot):
    """The yaw (radians about +Y) of the way the body faces, from root rotations (..., 3, 3)."""
    f = np.asarray(root_rot)[..., :, 2]
    return np.arctan2(f[..., 0], f[..., 2])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def forward_kinematics(rot, root, offsets, parents):
    """World positions (F, J, 3) and rotations (F, J, 3, 3) of every joint."""
    rot = np.asarray(rot, dtype=np.float64)
    F, J = rot.shape[:2]
    world_r = np.empty_like(rot)
    world_p = np.empty((F, J, 3))
    world_r[:, 0] = rot[:, 0]
    world_p[:, 0] = root
    for j in range(1, J):
        p = parents[j]
        world_r[:, j] = world_r[:, p] @ rot[:, j]
        world_p[:, j] = world_p[:, p] + world_r[:, p] @ np.asarray(offsets[j], dtype=np.float64)
    return world_p, world_r


def start_at_origin(rot, root):
    """Turn and move the whole clip so that its first frame stands at x = z = 0 facing +Z."""
    rot = np.array(rot, dtype=np.float64)
    root = np.array(root, dtype=np.float64)
    turn = rot_y(-heading(rot[0, 0]))
    centre = np.array([root[0, 0], 0.0, root[0, 2]])
    rot[:, 0] = turn @ rot[:, 0]
    root = (root - centre) @ turn.T
    return rot, root


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _body_pose(rot, root, offsets, parents):
    """Joint positions relative to the root, in the root's heading frame: the pose without where
    the body is or which way it faces."""
    pos, _ = forward_kinematics(rot, root, offsets, parents)
    rel = pos - root[:, None, :]
    out = np.empty_like(rel)
    for f in range(len(rel)):
        out[f] = rel[f] @ rot_y(-heading(rot[f, 0])).T
    return out


def loop_plan(frames, max_frames):
    """How many frames to generate for a loop of `frames` frames, and how many end frames to blend.

    The clip is made longer than asked: a lead-in before the loop starts (blended into its end, so
    the end runs into the start the way the motion ran into it) and a few spare frames, to choose
    the start whose pose matches the end best."""
    blend = int(np.clip(round(frames * 0.2), 6, 24))
    spare = int(max(0, min(15, max_frames - frames - blend)))
    return frames + blend + spare, blend


def make_loop(rot, root, frames, blend, offsets, parents):
    """Cut a loop of `frames` frames out of a longer clip whose last frame is its first frame.

    1. Choose where the loop starts: the start (after the lead-in) whose pose and joint speeds
       match those `frames - 1` frames later best.
    2. Take out the drift: the turn about +Y and the move over the ground between the loop's first
       and last frames, spread evenly over the clip, so it ends where it began, facing the same way.
    3. Blend the last frames into the frames that lead up to the start, ending exactly on it.
    """
    rot = np.array(rot, dtype=np.float64)
    root = np.array(root, dtype=np.float64)
    total = len(rot)
    assert total >= frames + blend and frames >= 2 and blend >= 2
    pose = _body_pose(rot, root, offsets, parents)
    vel = np.gradient(pose, axis=0)
    best, start = None, blend
    for s in range(blend, total - frames + 1):
        e = s + frames - 1
        cost = np.mean(np.linalg.norm(pose[s] - pose[e], axis=-1)) + 3.0 * np.mean(
            np.linalg.norm(vel[s] - vel[e], axis=-1))
        if best is None or cost < best:
            best, start = cost, s
    end = start + frames - 1

    turn = wrap(heading(rot[end, 0]) - heading(rot[start, 0]))
    centre = np.array([root[start, 0], 0.0, root[start, 2]])
    flat = np.array([1.0, 0.0, 1.0])
    shift = (rot_y(-turn) @ ((root[end] - centre) * flat)) * flat
    for f in range(total):
        s = (f - start) / (frames - 1)
        r = rot_y(-s * turn)
        rot[f, 0] = r @ rot[f, 0]
        h = (root[f] - centre) * flat
        root[f] = centre + r @ h - s * shift + np.array([0.0, root[f, 1], 0.0])

    q = continuous(mat_to_quat(rot))
    for m in range(blend):
        w, g = end - m, start - m
        t = _smoothstep(1.0 - m / (blend - 1))
        if t >= 1.0:
            q[w] = q[g]
            root[w] = root[g]
        else:
            q[w] = slerp(q[w], q[g], t)
            root[w] = (1.0 - t) * root[w] + t * root[g]
    return quat_to_mat(q[start:end + 1]), root[start:end + 1], start


def resample(q, root, count):
    """`count` frames spread evenly from the first frame to the last (both kept exactly)."""
    n = len(q)
    if count == n:
        return np.array(q), np.array(root)
    u = np.linspace(0.0, n - 1.0, count)
    i0 = np.clip(np.floor(u).astype(int), 0, n - 2)
    a = u - i0
    out_q = slerp(q[i0], q[i0 + 1], a[:, None])
    out_r = root[i0] * (1.0 - a)[:, None] + root[i0 + 1] * a[:, None]
    out_q[0], out_q[-1] = q[0], q[-1]
    out_r[0], out_r[-1] = root[0], root[-1]
    return out_q, out_r
