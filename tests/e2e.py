#!/usr/bin/env python3
"""End to end through hemera: the four test motions, their JSON checked, their GLB and BVH opened
in Blender, the time each takes, the engine's memory, and what the motions do badly.

    e2e.py --hemera http://127.0.0.1:12434 --model hy-motion-1.0 --out DIR \
           [--blender /media/shared/Linux/App/Blender/current/blender] [--render]

Writes DIR/<case>.json|glb|bvh, DIR/report.json, and with --render DIR/<case>.png (six moments of
each, front and side). Exits non-zero if a check failed.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from motion_math import forward_kinematics, quat_to_mat  # noqa: E402

CASES = [
    {"name": "wave", "prompt": "waves with the right hand", "duration": 4, "loop": False, "seed": 11},
    {"name": "twirl", "prompt": "a slow twirl", "duration": 5, "loop": False, "seed": 12},
    {"name": "sit", "prompt": "sits down", "duration": 4, "loop": False, "seed": 13},
    {"name": "idle_loop", "prompt": "stands relaxed, breathing calmly and shifting weight from one foot to the other",
     "duration": 4, "loop": True, "seed": 14},
]
FAILED = []


def check(cond, what, detail=""):
    print(("ok    " if cond else "FAIL  ") + what + ("" if cond or not detail else "\n      " + str(detail)[:400]),
          flush=True)
    if not cond:
        FAILED.append(what)


def call(base, method, path, body=None, raw=False):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            out = r.read()
            return r.status, (out if raw else json.loads(out)), dict(r.headers)
    except urllib.error.HTTPError as e:
        out = e.read()
        try:
            return e.code, json.loads(out), dict(e.headers)
        except ValueError:
            return e.code, out, dict(e.headers)


class Memory(threading.Thread):
    """Samples the motion engine's GPU memory (drm-total-gtt over its DRM clients, which counts
    ROCm's allocations) and its anonymous RSS, every 0.2 s."""

    def __init__(self):
        super().__init__(daemon=True)
        self.samples = []
        self.pid = None
        self.stop = False

    def find(self):
        for p in glob.glob("/proc/[0-9]*/cmdline"):
            try:
                cmd = open(p, "rb").read().split(b"\0")
            except OSError:
                continue
            if any(c.endswith(b"motion_server.py") for c in cmd):
                return int(p.split("/")[2])
        return None

    def read(self, pid):
        gtt, seen = 0, set()
        for fd in glob.glob("/proc/%d/fdinfo/*" % pid):
            try:
                info = open(fd).read()
            except OSError:
                continue
            if "drm-client-id" not in info:
                continue
            fields = dict(line.split(":", 1) for line in info.splitlines() if ":" in line)
            cid = fields["drm-client-id"].strip()
            if cid in seen:
                continue
            seen.add(cid)
            for key in ("drm-total-gtt", "drm-total-vram"):
                if key in fields:
                    v, unit = fields[key].split()
                    gtt += int(v) * {"KiB": 1024, "MiB": 1 << 20, "GiB": 1 << 30}.get(unit, 1)
        rss = 0
        for line in open("/proc/%d/status" % pid):
            if line.startswith("RssAnon:"):
                rss = int(line.split()[1]) * 1024
        return gtt, rss

    def run(self):
        while not self.stop:
            self.pid = self.pid if self.pid and os.path.exists("/proc/%d" % self.pid) else self.find()
            if self.pid:
                try:
                    self.samples.append((time.time(), *self.read(self.pid)))
                except OSError:
                    pass
            time.sleep(0.2)


def wait(base, gid, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st, d, _ = call(base, "GET", "/v1/motion/generations/" + gid)
        if st != 200 or d["status"] in ("completed", "failed", "cancelled"):
            return st, d, time.time() - t0
        time.sleep(0.25)
    return 0, {"status": "timeout"}, time.time() - t0


def analyse(doc):
    sk = doc["skeleton"]
    names = sk["joints"]
    ix = {n: i for i, n in enumerate(names)}
    q = np.array([f["rotations"] for f in doc["frames"]])
    root = np.array([f["translation"] for f in doc["frames"]])
    pos, _ = forward_kinematics(quat_to_mat(q), root, sk["offsets"], sk["parents"])
    fps = doc["fps"]
    out = {}
    # feet: each time a foot is planted (its toe within 2.5 cm of its lowest, hardly moving up or
    # down, for at least 5 frames), how far it moves over the ground before it lifts: it should not
    moved, worst, planted = 0.0, 0.0, 0
    for foot in ("L_Foot", "R_Foot"):
        p = pos[:, ix[foot]]
        y = p[:, 1]
        down = (y < y.min() + 0.025) & (np.abs(np.gradient(y)) * fps < 0.2)
        i = 0
        while i < len(down):
            if not down[i]:
                i += 1
                continue
            j = i
            while j < len(down) and down[j]:
                j += 1
            if j - i >= 5:
                d = float(np.linalg.norm(p[j - 1, [0, 2]] - p[i, [0, 2]]))
                moved, worst, planted = moved + d, max(worst, d), planted + (j - i)
            i = j
    out["planted_feet_slide"] = {"cm_per_s": round(100 * moved / max(planted / fps, 1e-9), 1),
                                 "worst_plant_cm": round(100 * worst, 1)}
    out["lowest_joint_m"] = round(float(pos[..., 1].min()), 3)
    head = pos[:, ix["Head"], 1]
    out["right_wrist_above_head"] = round(float(np.mean(pos[:, ix["R_Wrist"], 1] > head)), 2)
    out["left_wrist_above_head"] = round(float(np.mean(pos[:, ix["L_Wrist"], 1] > head)), 2)
    mid = len(pos) // 2
    out["mid_frame_right_wrist_minus_head_m"] = round(float(pos[mid, ix["R_Wrist"], 1] - head[mid]), 3)
    fwd = quat_to_mat(q[:, 0])[:, :, 2]
    yaw = np.unwrap(np.arctan2(fwd[:, 0], fwd[:, 2]))
    out["turned_degrees"] = round(float(np.degrees(yaw[-1] - yaw[0])), 1)
    out["pelvis_height_m"] = {"first": round(float(pos[0, 0, 1]), 3), "last": round(float(pos[-1, 0, 1]), 3),
                              "lowest": round(float(pos[:, 0, 1].min()), 3)}
    out["travelled_m"] = round(float(np.linalg.norm(root[-1, [0, 2]] - root[0, [0, 2]])), 3)
    # hands near the body: the closest either wrist comes to the spine and pelvis joints
    torso = [ix[n] for n in ("Pelvis", "Spine1", "Spine2", "Spine3", "Neck")]
    near = min(float(np.linalg.norm(pos[:, ix[w]][:, None] - pos[:, torso], axis=-1).min()) for w in ("L_Wrist", "R_Wrist"))
    out["closest_wrist_to_torso_joint_m"] = round(near, 3)
    return out, q, root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hemera", default="http://127.0.0.1:12434")
    ap.add_argument("--model", default="hy-motion-1.0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--blender", default="/media/shared/Linux/App/Blender/current/blender")
    ap.add_argument("--render", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    base = a.hemera.rstrip("/")
    mem = Memory()
    mem.start()
    report = {"model": a.model, "cases": {}}

    st, d, _ = call(base, "GET", "/v1/models")
    entry = next((m for m in d.get("data", []) if m["id"] == a.model), None)
    check(entry is not None and entry["hemera"]["kind"] == "motion", "hemera lists %s as a motion model" % a.model, entry)

    # the first request starts the engine: retried while hemera answers 503 (loading)
    t0 = time.time()
    for _ in range(120):
        st, d, h = call(base, "POST", "/v1/motion/generations",
                        {"model": a.model, "prompt": "walks forward", "duration": 2, "seed": 1})
        if st != 503:
            break
        time.sleep(float(h.get("Retry-After", 5)))
    st, d, _ = wait(base, d["id"]) if st == 200 else (st, d, 0)
    report["cold_start_s"] = round(time.time() - t0, 1)
    check(d.get("status") == "completed", "cold start: the first motion is made (%.0f s with loading)" % report["cold_start_s"], d)
    call(base, "DELETE", "/v1/motion/generations/" + d.get("id", "x"))
    idle = [s for s in mem.samples if s[0] > time.time() - 1.0]
    report["memory_loaded_gb"] = {"gpu": round(idle[-1][1] / 1e9, 2), "rss": round(idle[-1][2] / 1e9, 2)} if idle else None

    for c in CASES:
        print("\n== %s (%s s%s)" % (c["name"], c["duration"], ", loop" if c["loop"] else ""), flush=True)
        t_start = time.time()
        st, d, _ = call(base, "POST", "/v1/motion/generations", {"model": a.model, "prompt": c["prompt"],
                                                                 "duration": c["duration"], "loop": c["loop"],
                                                                 "seed": c["seed"]})
        check(st == 200 and d.get("status") in ("queued", "in_progress"), "POST answers at once with an id", (st, d))
        st, d, took = wait(base, d["id"])
        check(d.get("status") == "completed", "made in %.1f s" % took, d)
        gid = d["id"]
        st, doc, _ = call(base, "GET", "/v1/motion/generations/%s/content?format=json" % gid)
        n = int(c["duration"] * 30)
        sk = doc["skeleton"]
        check(len(sk["joints"]) == 22 and len(sk["parents"]) == 22 and len(sk["offsets"]) == 22,
              "22 joints, with parents and offsets")
        check(doc["fps"] == 30 and doc["frame_count"] == n and len(doc["frames"]) == n,
              "frames = duration x fps (%d)" % n, doc["frame_count"])
        norms = np.linalg.norm(np.array([f["rotations"] for f in doc["frames"]]), axis=-1)
        check(np.abs(norms - 1).max() < 1e-4 and norms.shape[1] == 22,
              "unit quaternions (worst %.1e)" % np.abs(norms - 1).max())
        info, q, root = analyse(doc)
        if c["name"] == "wave":
            check(info["mid_frame_right_wrist_minus_head_m"] > 0,
                  "mid-wave the right hand is above the head (%.2f m)" % info["mid_frame_right_wrist_minus_head_m"], info)
        if c["loop"]:
            same = np.all(np.abs(np.abs(np.sum(q[0] * q[-1], -1)) - 1) < 1e-5) and np.allclose(root[0], root[-1], atol=1e-4)
            check(bool(same), "the loop's last frame is its first frame")
        for fmt in ("json", "glb", "bvh"):
            st, data, hdr = call(base, "GET", "/v1/motion/generations/%s/content?format=%s" % (gid, fmt), raw=True)
            with open(os.path.join(a.out, c["name"] + "." + fmt), "wb") as f:
                f.write(data)
        check(open(os.path.join(a.out, c["name"] + ".glb"), "rb").read(4) == b"glTF", "GLB fetched")
        st, d2, _ = call(base, "DELETE", "/v1/motion/generations/" + gid)
        st2, _, _ = call(base, "GET", "/v1/motion/generations/" + gid)
        st3, _, _ = call(base, "GET", "/v1/motion/generations/%s/content?format=json" % gid)
        check(st == 200 and st2 == 404 and st3 == 404, "after DELETE nothing of it is left")
        window = [s for s in mem.samples if t_start <= s[0] <= time.time()]
        report["cases"][c["name"]] = dict(info, seconds=round(took, 1), timing_ms=d.get("timing_ms"),
                                          peak_gpu_gb=round(max(s[1] for s in window) / 1e9, 2) if window else None,
                                          peak_rss_gb=round(max(s[2] for s in window) / 1e9, 2) if window else None)
        print("      " + json.dumps(report["cases"][c["name"]]), flush=True)

    print("\n== requests that must fail", flush=True)
    for body, what in (({"model": a.model}, "no prompt"), ({"model": a.model, "prompt": "waves", "duration": 11}, "11 s"),
                       ({"model": a.model, "prompt": "waves", "constraints": [{"frame": 0}]}, "constraints"),
                       ({"model": a.model, "prompt": "waves", "fps": 0}, "fps 0")):
        st, d, _ = call(base, "POST", "/v1/motion/generations", body)
        check(st == 400 and isinstance(d, dict) and isinstance(d.get("error"), str), "400 for %s: %s" % (what, d.get("error")))

    print("\n== Blender", flush=True)
    files = [os.path.join(a.out, c["name"] + ".json") for c in CASES]
    cmd = [a.blender, "-b", "--factory-startup", "--python",
           os.path.join(os.path.dirname(os.path.abspath(__file__)), "blender_check.py"), "--"]
    cmd += (["--render", a.out] if a.render else []) + files
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    results = [json.loads(line[7:]) for line in res.stdout.splitlines() if line.startswith("RESULT ")]
    check(len(results) == len(CASES), "Blender opened every file", res.stderr[-800:])
    for r in results:
        check(r["glb"]["ok"], "Blender: %s.glb is a 22-bone armature with %s frames, posed as the JSON says (within %.1e m)"
              % (r["name"], r["glb"].get("frames"), r["glb"].get("max_offset_m") or -1), r["glb"])
        check(r.get("bvh", {}).get("ok"), "Blender: %s.bvh likewise (within %.1e m)" % (
            r["name"], r.get("bvh", {}).get("max_offset_m") or -1), r.get("bvh"))
    report["blender"] = results
    mem.stop = True
    report["memory_peak_gb"] = {"gpu": round(max(s[1] for s in mem.samples) / 1e9, 2),
                                "rss": round(max(s[2] for s in mem.samples) / 1e9, 2)} if mem.samples else None
    report["failed"] = FAILED
    with open(os.path.join(a.out, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    print("\nmemory loaded %s, peak %s; cold start %.0f s" % (report["memory_loaded_gb"], report["memory_peak_gb"],
                                                             report["cold_start_s"]))
    print("%d failed" % len(FAILED) if FAILED else "all passed")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
