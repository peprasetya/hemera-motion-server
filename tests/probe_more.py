"""Loop seams, a reload from the page cache, and a 10 s clip at 60 fps, through hemera."""
import json, sys, time, urllib.request
import numpy as np
sys.path.insert(0, __file__.rsplit("/tests/", 1)[0])
from motion_math import forward_kinematics, quat_to_mat
BASE = sys.argv[1]
def call(m, p, b=None, raw=False):
    r = urllib.request.Request(BASE + p, data=json.dumps(b).encode() if b else None, method=m, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=300) as f:
            d = f.read(); return f.status, (d if raw else json.loads(d))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())
def make(item):
    t0 = time.time()
    while True:
        st, job = call("POST", "/v1/motion/generations", dict(model="hy-motion-1.0", **item))
        if st != 503: break
        time.sleep(2)
    while job["status"] in ("queued", "in_progress"):
        time.sleep(0.2); st, job = call("GET", "/v1/motion/generations/" + job["id"])
    took = time.time() - t0
    out = {f: call("GET", "/v1/motion/generations/%s/content?format=%s" % (job["id"], f), raw=True)[1] for f in ("json", "glb", "bvh")}
    call("DELETE", "/v1/motion/generations/" + job["id"])
    return took, json.loads(out["json"]), {f: len(v) for f, v in out.items()}, job
def seam(d):
    sk = d["skeleton"]
    q = np.array([f["rotations"] for f in d["frames"]]); r = np.array([f["translation"] for f in d["frames"]])
    pos, _ = forward_kinematics(quat_to_mat(q), r, sk["offsets"], sk["parents"])
    step = np.linalg.norm(np.diff(pos, axis=0), axis=-1).max(axis=1)  # largest joint move per frame
    # played as a loop: ... L-3, L-2, L-1 (= 0), 1, 2 ...; the wrap is the step L-2 -> L-1 then 0 -> 1
    acc = np.linalg.norm(pos[-2] - 2 * pos[-1] + pos[1], axis=-1).max()  # second difference across the wrap
    accs = np.linalg.norm(pos[:-2] - 2 * pos[1:-1] + pos[2:], axis=-1).max(axis=1)
    return "largest joint step %.1f cm/frame (median %.1f); across the wrap: steps %.1f and %.1f cm, acceleration %.2f cm/frame² (clip p95 %.2f, max %.2f)" % (
        100 * step.max(), 100 * np.median(step), 100 * step[-1], 100 * step[0], 100 * acc, 100 * np.percentile(accs, 95), 100 * accs.max())
for item in ({"prompt": "stands relaxed, breathing calmly and shifting weight from one foot to the other", "duration": 4, "seed": 14, "loop": True},
             {"prompt": "stands idle, looking around", "duration": 6, "seed": 5, "loop": True},
             {"prompt": "dances happily in place", "duration": 3, "seed": 9, "loop": True}):
    took, d, sizes, job = make(item)
    print("loop  %-40s %s" % (item["prompt"][:40], seam(d)), flush=True)
st, _ = call("POST", "/hemera/api/models/stop", {"name": "hy-motion-1.0"})
time.sleep(3)
took, d, sizes, job = make({"prompt": "walks forward", "duration": 2, "seed": 1})
print("reload from page cache, then a 2 s clip: %.1f s" % took, flush=True)
took, d, sizes, job = make({"prompt": "a person does a cartwheel", "duration": 10, "fps": 60, "seed": 3})
print("10 s at 60 fps: %.1f s, %d frames, sizes %s, timing %s" % (took, d["frame_count"], sizes, job.get("timing_ms")), flush=True)
took, d, sizes, job = make({"prompt": "waves with the right hand", "duration": 1, "seed": 3})
print("1 s at 30 fps: %.1f s, %d frames, sizes %s" % (took, d["frame_count"], sizes), flush=True)
