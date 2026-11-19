"""A finished clip as JSON (what clients read), binary glTF (an animated skeleton, for Blender and
glTF viewers) or BVH. All three carry the same skeleton and frames; none carries the prompt."""
import json
import struct

import numpy as np
from scipy.spatial.transform import Rotation

from motion_math import forward_kinematics, quat_to_mat


class Clip:
    def __init__(self, model, skeleton, quats, root, fps, loop, seed):
        self.model = model
        self.skeleton = skeleton  # {"joints", "parents", "offsets", "ends"}
        self.quats = np.asarray(quats, dtype=np.float64)  # (F, J, 4) x y z w
        self.root = np.asarray(root, dtype=np.float64)  # (F, 3) metres
        self.fps = fps
        self.loop = loop
        self.seed = seed

    @property
    def frames(self):
        return len(self.quats)


def _r(v, digits):
    return [round(float(x), digits) for x in v]


def to_json(clip):
    sk = clip.skeleton
    doc = {
        "object": "motion",
        "model": clip.model,
        "fps": clip.fps,
        "frame_count": clip.frames,
        "duration": round(clip.frames / clip.fps, 4),
        "loop": clip.loop,
        "seed": clip.seed,
        "skeleton": {
            "joints": list(sk["joints"]),
            "parents": [int(p) for p in sk["parents"]],
            "offsets": [_r(o, 5) for o in sk["offsets"]],
            "units": "metres",
            "up": "+Y",
            "forward": "+Z",
            "left": "+X",
        },
        "frames": [{"translation": _r(clip.root[f], 5), "rotations": [_r(q, 6) for q in clip.quats[f]]}
                   for f in range(clip.frames)],
    }
    return json.dumps(doc, separators=(",", ":")).encode()


# ---- binary glTF -----------------------------------------------------------------------------

def _bone_mesh(rest, parents, ends):
    """A thin, flat-shaded octahedron along every bone (and to each end site), each bound wholly
    to the joint the bone starts at: enough to see the motion in any glTF viewer."""
    pos, nrm, joint = [], [], []
    segments = [(parents[j], rest[parents[j]], rest[j]) for j in range(1, len(parents))]
    segments += [(j, rest[j], rest[j] + np.asarray(off)) for j, off in ends.items()]
    for j, a, b in segments:
        axis = b - a
        length = np.linalg.norm(axis)
        if length < 1e-6:
            continue
        d = axis / length
        side = np.cross(d, [0.0, 0.0, 1.0] if abs(d[2]) < 0.9 else [1.0, 0.0, 0.0])
        side /= np.linalg.norm(side)
        up = np.cross(d, side)
        w = 0.06 * length + 0.008
        mid = a + d * length * 0.2
        ring = [mid + w * side, mid + w * up, mid - w * side, mid - w * up]
        for i in range(4):
            r0, r1 = ring[i], ring[(i + 1) % 4]
            for tri in ((a, r1, r0), (b, r0, r1)):
                n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
                n /= np.linalg.norm(n) or 1.0
                for v in tri:
                    pos.append(v)
                    nrm.append(n)
                    joint.append(j)
    return np.array(pos, np.float32), np.array(nrm, np.float32), np.array(joint, np.uint16)


def to_glb(clip):
    sk = clip.skeleton
    parents = [int(p) for p in sk["parents"]]
    offsets = np.asarray(sk["offsets"], dtype=np.float64)
    J = len(parents)
    rest, _ = forward_kinematics(np.tile(np.eye(3), (1, J, 1, 1)), offsets[0][None], offsets, parents)
    rest = rest[0]

    blob = bytearray()
    views, accessors = [], []

    def add(array, component, kind, target=None, minmax=False):
        array = np.ascontiguousarray(array)
        while len(blob) % 4:
            blob.append(0)
        view = {"buffer": 0, "byteOffset": len(blob), "byteLength": array.nbytes}
        if target:
            view["target"] = target
        blob.extend(array.tobytes())
        views.append(view)
        acc = {"bufferView": len(views) - 1, "componentType": component, "count": int(array.shape[0]), "type": kind}
        if minmax:
            flat = array.reshape(array.shape[0], -1)
            acc["min"] = [float(x) for x in flat.min(axis=0)]
            acc["max"] = [float(x) for x in flat.max(axis=0)]
        accessors.append(acc)
        return len(accessors) - 1

    FLOAT, USHORT = 5126, 5123
    pos, nrm, joint = _bone_mesh(rest, parents, {int(k): v for k, v in sk["ends"].items()})
    a_pos = add(pos, FLOAT, "VEC3", 34962, minmax=True)
    a_nrm = add(nrm, FLOAT, "VEC3", 34962)
    joints4 = np.zeros((len(joint), 4), np.uint16)
    joints4[:, 0] = joint
    weights4 = np.zeros((len(joint), 4), np.float32)
    weights4[:, 0] = 1.0
    a_jnt = add(joints4, USHORT, "VEC4", 34962)
    a_wgt = add(weights4, FLOAT, "VEC4", 34962)
    ibm = np.tile(np.eye(4, dtype=np.float32), (J, 1, 1))
    ibm[:, :3, 3] = -rest
    a_ibm = add(np.transpose(ibm, (0, 2, 1)).reshape(J, 16), FLOAT, "MAT4")  # glTF matrices are column-major

    times = np.arange(clip.frames, dtype=np.float32) / np.float32(clip.fps)
    a_time = add(times, FLOAT, "SCALAR", minmax=True)
    samplers, channels = [], []
    a_tr = add(clip.root.astype(np.float32), FLOAT, "VEC3")
    samplers.append({"input": a_time, "output": a_tr, "interpolation": "LINEAR"})
    channels.append({"sampler": 0, "target": {"node": 0, "path": "translation"}})
    for j in range(J):
        a_rot = add(clip.quats[:, j].astype(np.float32), FLOAT, "VEC4")
        samplers.append({"input": a_time, "output": a_rot, "interpolation": "LINEAR"})
        channels.append({"sampler": len(samplers) - 1, "target": {"node": j, "path": "rotation"}})

    nodes = []
    for j in range(J):
        node = {"name": sk["joints"][j], "translation": [float(x) for x in offsets[j]]}
        kids = [k for k in range(J) if parents[k] == j]
        if kids:
            node["children"] = kids
        nodes.append(node)
    nodes.append({"name": "Body", "mesh": 0, "skin": 0})
    doc = {
        "asset": {"version": "2.0", "generator": "motion-server (" + clip.model + ")"},
        "extras": {"fps": clip.fps, "loop": clip.loop, "units": "metres"},
        "scene": 0,
        "scenes": [{"nodes": [0, J]}],
        "nodes": nodes,
        "skins": [{"name": "Skeleton", "joints": list(range(J)), "skeleton": 0, "inverseBindMatrices": a_ibm}],
        "meshes": [{"name": "Bones", "primitives": [{"attributes": {"POSITION": a_pos, "NORMAL": a_nrm,
                                                                    "JOINTS_0": a_jnt, "WEIGHTS_0": a_wgt},
                                                     "material": 0}]}],
        "materials": [{"name": "Bone", "pbrMetallicRoughness": {"baseColorFactor": [0.8, 0.78, 0.72, 1.0],
                                                                 "metallicFactor": 0.0, "roughnessFactor": 0.7}}],
        "animations": [{"name": "motion", "samplers": samplers, "channels": channels}],
        "buffers": [{"byteLength": len(blob)}],
        "bufferViews": views,
        "accessors": accessors,
    }
    js = json.dumps(doc, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    blob.extend(b"\0" * (-len(blob) % 4))
    total = 12 + 8 + len(js) + 8 + len(blob)
    return (struct.pack("<4sII", b"glTF", 2, total) + struct.pack("<I4s", len(js), b"JSON") + js
            + struct.pack("<I4s", len(blob), b"BIN\0") + bytes(blob))


# ---- BVH ---------------------------------------------------------------------------------------

def to_bvh(clip):
    """BVH in metres, Y up. The root has no offset: its channels are its position in the world."""
    sk = clip.skeleton
    names, parents = sk["joints"], [int(p) for p in sk["parents"]]
    offsets = np.asarray(sk["offsets"], dtype=np.float64)
    ends = {int(k): v for k, v in sk["ends"].items()}
    J = len(names)
    lines = ["HIERARCHY"]
    order = []  # the frames list the channels in the order the joints appear in the hierarchy

    def emit(j, depth):
        order.append(j)
        pad = "  " * depth
        if j == 0:
            lines.append("ROOT " + names[j])
            lines.append("{")
            lines.append(pad + "  OFFSET 0.000000 0.000000 0.000000")
            lines.append(pad + "  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation")
        else:
            lines.append(pad + "JOINT " + names[j])
            lines.append(pad + "{")
            lines.append(pad + "  OFFSET %.6f %.6f %.6f" % tuple(offsets[j]))
            lines.append(pad + "  CHANNELS 3 Zrotation Xrotation Yrotation")
        kids = [k for k in range(J) if parents[k] == j]
        for k in kids:
            emit(k, depth + 1)
        if not kids:
            e = ends.get(j, [0.0, 0.05, 0.0])
            lines.append(pad + "  End Site")
            lines.append(pad + "  {")
            lines.append(pad + "    OFFSET %.6f %.6f %.6f" % tuple(e))
            lines.append(pad + "  }")
        lines.append(pad + "}")

    emit(0, 0)
    # R = Rz * Rx * Ry, the order the channels are listed in
    euler = Rotation.from_matrix(quat_to_mat(clip.quats).reshape(-1, 3, 3)).as_euler("ZXY", degrees=True)
    euler = np.degrees(np.unwrap(np.radians(euler.reshape(clip.frames, J, 3)), axis=0))
    lines += ["MOTION", "Frames: %d" % clip.frames, "Frame Time: %.8f" % (1.0 / clip.fps)]
    for f in range(clip.frames):
        vals = ["%.6f" % v for v in clip.root[f]]
        for j in order:
            vals += ["%.4f" % v for v in euler[f, j]]
        lines.append(" ".join(vals))
    return ("\n".join(lines) + "\n").encode()
