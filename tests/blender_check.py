"""Run inside Blender: open each motion's GLB (and BVH), and check that Blender sees a 22-bone
armature with the right number of frames, posed where the JSON says.

    blender -b --factory-startup --python tests/blender_check.py -- [--render DIR] NAME.json ...

Each NAME.json needs NAME.glb (and optionally NAME.bvh) beside it. Prints one "RESULT {json}" line
per file. With --render, writes DIR/NAME.png: the clip at six moments, from the front and the side.
"""
import json
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector


def fk(doc, f):
    sk = doc["skeleton"]
    fr = doc["frames"][f]
    pos, rot = [], []
    for j, p in enumerate(sk["parents"]):
        x, y, z, w = fr["rotations"][j]
        q = Quaternion((w, x, y, z)).to_matrix()
        if p < 0:
            rot.append(q)
            pos.append(Vector(fr["translation"]))
        else:
            rot.append(rot[p] @ q)
            pos.append(pos[p] + rot[p] @ Vector(sk["offsets"][j]))
    return pos


def to_blender(v):  # glTF and BVH here are +Y up; Blender is +Z up
    return Vector((v[0], -v[2], v[1]))


def clear():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def armature():
    arms = [o for o in bpy.context.scene.objects if o.type == "ARMATURE"]
    return arms[0] if arms else None


def compare(doc, arm, first_frame, names):
    """Largest distance between Blender's posed bone heads and the JSON's joints, over a few frames."""
    worst = 0.0
    n = doc["frame_count"]
    for f in sorted({0, n // 4, n // 2, (3 * n) // 4, n - 1}):
        bpy.context.scene.frame_set(first_frame + f)
        want = fk(doc, f)
        for j, name in enumerate(names):
            pb = arm.pose.bones.get(name)
            if pb is None:
                return None
            got = arm.matrix_world @ pb.head
            worst = max(worst, (got - to_blender(want[j])).length)
    return worst


def check_glb(path, doc):
    clear()
    bpy.context.scene.render.fps = doc["fps"]
    bpy.ops.import_scene.gltf(filepath=path)
    arm = armature()
    if arm is None:
        return {"ok": False, "why": "no armature"}
    bones = len(arm.data.bones)
    action = arm.animation_data.action if arm.animation_data else None
    if action is None:
        return {"ok": False, "why": "no animation", "bones": bones}
    lo, hi = action.frame_range
    frames = int(round(hi - lo)) + 1
    off = compare(doc, arm, int(round(lo)), doc["skeleton"]["joints"])
    ok = bones == 22 and frames == doc["frame_count"] and off is not None and off < 0.002
    return {"ok": ok, "bones": bones, "frames": frames, "max_offset_m": off}


def check_bvh(path, doc):
    clear()
    bpy.context.scene.render.fps = doc["fps"]
    bpy.ops.import_anim.bvh(filepath=path, global_scale=1.0, frame_start=1, use_fps_scale=False,
                            rotate_mode="NATIVE", axis_forward="-Z", axis_up="Y")
    arm = armature()
    if arm is None:
        return {"ok": False, "why": "no armature"}
    action = arm.animation_data.action
    lo, hi = action.frame_range
    off = compare(doc, arm, int(round(lo)), doc["skeleton"]["joints"])
    frames = int(round(hi - lo)) + 1
    ok = len(arm.data.bones) == 22 and frames == doc["frame_count"] and off is not None and off < 0.002
    return {"ok": ok, "bones": len(arm.data.bones), "frames": frames, "max_offset_m": off}


def render(path, doc, out):
    """Six moments, front (top row) and side (bottom row), to see what the model made."""
    clear()
    scene = bpy.context.scene
    scene.render.fps = doc["fps"]
    bpy.ops.import_scene.gltf(filepath=path)
    arm = armature()
    lo = int(round(arm.animation_data.action.frame_range[0]))
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "OBJECT"
    scene.render.resolution_x = scene.render.resolution_y = 300
    scene.render.film_transparent = False
    scene.world = bpy.data.worlds.new("w")
    scene.world.color = (0.95, 0.95, 0.95)
    bpy.ops.mesh.primitive_plane_add(size=6)  # the ground
    bpy.context.object.color = (0.6, 0.65, 0.7, 1)
    for o in scene.objects:
        if o.type == "MESH" and o.parent == arm or (o.type == "MESH" and o.find_armature()):
            o.color = (0.85, 0.45, 0.2, 1)
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    scene.collection.objects.link(cam)
    scene.camera = cam
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = 2.6
    n = doc["frame_count"]
    moments = [int(round(k * (n - 1) / 5)) for k in range(6)]
    tiles = []
    for view in ("front", "side"):
        row = []
        for f in moments:
            scene.frame_set(lo + f)
            hips = arm.matrix_world @ arm.pose.bones["Pelvis"].head
            if view == "front":  # the body faces glTF +Z, Blender -Y
                cam.location = (hips.x, hips.y - 6, 0.95)
                cam.rotation_euler = (math.pi / 2, 0, 0)
            else:
                cam.location = (hips.x + 6, hips.y, 0.95)
                cam.rotation_euler = (math.pi / 2, 0, math.pi / 2)
            scene.render.filepath = out + ".tile.png"
            bpy.ops.render.render(write_still=True)
            img = bpy.data.images.load(out + ".tile.png")
            px = np.array(img.pixels[:], dtype=np.float32).reshape(300, 300, 4)
            bpy.data.images.remove(img)
            row.append(px)
        tiles.append(np.concatenate(row, axis=1))
    sheet = np.concatenate(tiles[::-1], axis=0)  # Blender images start at the bottom row
    h, w = sheet.shape[:2]
    img = bpy.data.images.new("sheet", w, h)
    img.pixels = sheet.ravel()
    img.filepath_raw = out
    img.file_format = "PNG"
    img.save()
    os.remove(out + ".tile.png")


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    render_dir = None
    if argv[:1] == ["--render"]:
        render_dir, argv = argv[1], argv[2:]
    for js in argv:
        stem = js[:-5]
        with open(js) as f:
            doc = json.load(f)
        res = {"name": os.path.basename(stem), "glb": check_glb(stem + ".glb", doc)}
        if os.path.exists(stem + ".bvh"):
            res["bvh"] = check_bvh(stem + ".bvh", doc)
        if render_dir:
            render(stem + ".glb", doc, os.path.join(render_dir, os.path.basename(stem) + ".png"))
            res["render"] = os.path.join(render_dir, os.path.basename(stem) + ".png")
        print("RESULT " + json.dumps(res), flush=True)


main()
