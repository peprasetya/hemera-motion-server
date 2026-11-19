"""Checks of the maths and the three formats on made-up motion; no model or GPU needed.

    python -I tests/test_offline.py
"""
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exports import Clip, to_bvh, to_glb, to_json  # noqa: E402
from motion_math import (continuous, forward_kinematics, heading, loop_plan, make_loop, mat_to_quat,  # noqa: E402
                         quat_to_mat, resample, slerp, start_at_origin)

NAMES = ["Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2", "L_Ankle", "R_Ankle", "Spine3", "L_Foot",
         "R_Foot", "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow", "L_Wrist",
         "R_Wrist"]
PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
OFFSETS = [[-0.002, 0.97, 0.028], [0.082, -0.09, -0.003], [-0.079, -0.09, -0.003], [0.002, 0.094, 0.004],
           [0.019, -0.377, -0.01], [-0.017, -0.377, -0.01], [0.0, 0.123, -0.003], [0.001, -0.398, -0.052],
           [0.001, -0.398, -0.052], [0.0, 0.08, -0.024], [0.013, -0.072, 0.143], [-0.012, -0.072, 0.143],
           [0.0, 0.196, -0.02], [0.055, 0.119, -0.025], [-0.055, 0.119, -0.025], [0.0, 0.096, -0.002],
           [0.111, 0.012, 0.003], [-0.111, 0.012, 0.003], [0.268, 0.012, -0.013], [-0.268, 0.012, -0.013],
           [0.269, 0.004, -0.013], [-0.269, 0.004, -0.013]]
ENDS = {15: [0, 0.172, 0], 20: [0.09, 0, 0], 21: [-0.09, 0, 0], 10: [0, 0, 0.055], 11: [0, 0, 0.055]}
SK = {"joints": NAMES, "parents": PARENTS, "offsets": OFFSETS, "ends": ENDS}
FAILED = []


def check(cond, what):
    print(("ok    " if cond else "FAIL  ") + what)
    if not cond:
        FAILED.append(what)


def same_rotation(a, b, tol=1e-6):
    return np.all(np.abs(np.abs(np.sum(a * b, axis=-1)) - 1) < tol)


def made_up(frames, seed=0):
    """A wobbling, turning, drifting body: rotations from smooth random axis-angles."""
    from scipy.spatial.transform import Rotation
    rng = np.random.default_rng(seed)
    t = np.arange(frames) / 30.0
    amp, freq, phase = rng.normal(0, 0.4, (22, 3)), rng.uniform(0.2, 1.5, (22, 3)), rng.uniform(0, 6, (22, 3))
    rv = amp[None] * np.sin(2 * np.pi * freq[None] * t[:, None, None] + phase[None])
    rv[:, 0] = 0
    rv[:, 0, 1] = 0.9 * t + 0.3  # turning about +Y
    R = Rotation.from_rotvec(rv.reshape(-1, 3)).as_matrix().reshape(frames, 22, 3, 3)
    root = np.stack([0.4 * t + 0.1, 0.95 + 0.03 * np.sin(3 * t), 0.2 * t - 0.3], -1)
    return R, root


def main():
    R, root = made_up(150)
    q = mat_to_quat(R)
    check(np.allclose(quat_to_mat(q), R, atol=1e-9), "matrix -> quaternion -> matrix")
    qc = continuous(-q * np.sign(np.random.default_rng(1).normal(size=q.shape[:2]))[..., None])
    check(np.all(np.sum(qc[1:] * qc[:-1], -1) >= 0), "continuous: neighbouring frames on the same side")
    check(np.allclose(slerp(q[0], q[1], 0.0), q[0]) and np.allclose(slerp(q[0], q[1], 1.0), q[1]), "slerp ends")

    R0, r0 = start_at_origin(R, root)
    check(abs(r0[0, 0]) < 1e-12 and abs(r0[0, 2]) < 1e-12 and abs(heading(R0[0, 0])) < 1e-9,
          "start_at_origin: first frame at x = z = 0 facing +Z")
    p_before, _ = forward_kinematics(R, root, OFFSETS, PARENTS)
    p_after, _ = forward_kinematics(R0, r0, OFFSETS, PARENTS)
    d_before = np.linalg.norm(p_before[:, 20] - p_before[:, 7], axis=-1)
    d_after = np.linalg.norm(p_after[:, 20] - p_after[:, 7], axis=-1)
    check(np.allclose(d_before, d_after), "start_at_origin moves the clip rigidly")

    frames = 120
    total, blend = loop_plan(frames, 360)
    R, root = made_up(total, seed=3)
    LR, Lroot, start = make_loop(R, root, frames, blend, OFFSETS, PARENTS)
    check(len(LR) == frames and len(Lroot) == frames, "loop has the frames asked for (%d, start %d)" % (frames, start))
    lq = continuous(mat_to_quat(LR))
    check(same_rotation(lq[0], lq[-1]) and np.allclose(Lroot[0], Lroot[-1]), "loop: last frame is the first frame")
    jumps = np.linalg.norm(np.diff(Lroot, axis=0), axis=-1)
    check(jumps.max() < 0.1, "loop: no jump in the root path (largest step %.3f m)" % jumps.max())

    q30, r30 = continuous(mat_to_quat(LR)), Lroot
    q60, r60 = resample(q30, r30, 240)
    check(len(q60) == 240 and same_rotation(q60[0], q60[-1]) and np.allclose(r60[0], r60[-1]),
          "resample to 60 fps keeps the frame count and the loop")
    check(np.allclose(np.linalg.norm(q60, axis=-1), 1), "resampled quaternions are unit")

    clip = Clip("test-model", SK, q30, r30, 30, True, 7)
    doc = json.loads(to_json(clip))
    check(doc["frame_count"] == frames and len(doc["frames"]) == frames, "json: frame count")
    check(len(doc["skeleton"]["joints"]) == 22 and len(doc["frames"][0]["rotations"]) == 22, "json: 22 joints")
    norms = np.linalg.norm(np.array([f["rotations"] for f in doc["frames"]]), axis=-1)
    check(np.abs(norms - 1).max() < 1e-5, "json: unit quaternions (worst %.1e)" % np.abs(norms - 1).max())
    check("prompt" not in json.dumps(doc), "json: no prompt")

    glb = to_glb(clip)
    magic, version, length = struct.unpack_from("<4sII", glb, 0)
    jlen, jtype = struct.unpack_from("<I4s", glb, 12)
    gl = json.loads(glb[20:20 + jlen])
    blen, btype = struct.unpack_from("<I4s", glb, 20 + jlen)
    check(magic == b"glTF" and version == 2 and length == len(glb) and jtype == b"JSON" and btype == b"BIN\0",
          "glb: header and chunks")
    check(gl["buffers"][0]["byteLength"] == blen, "glb: buffer length")
    check(all(v["byteOffset"] + v["byteLength"] <= blen and v["byteOffset"] % 4 == 0 for v in gl["bufferViews"]),
          "glb: buffer views inside the buffer, aligned")
    anim = gl["animations"][0]
    check(len(anim["channels"]) == 23 and sum(c["target"]["path"] == "rotation" for c in anim["channels"]) == 22,
          "glb: one translation and 22 rotation channels")
    check(all(gl["accessors"][s["input"]]["count"] == frames for s in anim["samplers"]), "glb: every sampler has the frames")
    check(len(gl["skins"][0]["joints"]) == 22 and gl["nodes"][22]["skin"] == 0, "glb: a skin over the 22 joints")

    bvh = to_bvh(clip).decode()
    head, motion = bvh.split("MOTION\n")
    check(head.count("JOINT ") == 21 and head.count("ROOT ") == 1 and head.count("End Site") == 5, "bvh: hierarchy")
    rows = motion.strip().split("\n")[2:]
    check(len(rows) == frames and all(len(r.split()) == 6 + 3 * 21 for r in rows), "bvh: frames and channels")

    print("\n%d failed" % len(FAILED) if FAILED else "\nall passed")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
