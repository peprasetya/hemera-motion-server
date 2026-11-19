"""Try prompts and seeds through hemera and print what came out: how far the body turned, how far
it went, how low the hips got, the right hand's best height over the head, and foot drift."""
import json, sys, time, urllib.request
import numpy as np
sys.path.insert(0, __file__.rsplit("/tests/", 1)[0])
from motion_math import forward_kinematics, quat_to_mat
BASE = sys.argv[1]
def call(m, p, b=None):
    r = urllib.request.Request(BASE + p, data=json.dumps(b).encode() if b else None, method=m, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=120) as f: return json.loads(f.read())
def drift(pos, ix):
    """Feet: contact intervals (toe within 2.5 cm of its lowest, hardly moving up or down, >= 5 frames);
    how far a planted foot moves over the ground, per second planted."""
    worst, moved, planted = 0.0, 0.0, 0
    for foot in ("L_Foot", "R_Foot"):
        p = pos[:, ix[foot]]; y = p[:, 1]; vy = np.abs(np.gradient(y)) * 30
        down = (y < y.min() + 0.025) & (vy < 0.2)
        i = 0
        while i < len(down):
            if not down[i]: i += 1; continue
            j = i
            while j < len(down) and down[j]: j += 1
            if j - i >= 5:
                d = float(np.linalg.norm(p[j - 1, [0, 2]] - p[i, [0, 2]]))
                worst, moved, planted = max(worst, d), moved + d, planted + (j - i)
            i = j
    return round(100 * moved / max(planted / 30, 1e-9), 1), round(100 * worst, 1)
for item in json.loads(sys.argv[2]):
    t0 = time.time()
    job = call("POST", "/v1/motion/generations", dict(model="hy-motion-1.0", **item))
    while job["status"] in ("queued", "in_progress"):
        time.sleep(0.3); job = call("GET", "/v1/motion/generations/" + job["id"])
    d = call("GET", "/v1/motion/generations/%s/content" % job["id"]); call("DELETE", "/v1/motion/generations/" + job["id"])
    sk = d["skeleton"]; ix = {k: i for i, k in enumerate(sk["joints"])}
    q = np.array([f["rotations"] for f in d["frames"]]); r = np.array([f["translation"] for f in d["frames"]])
    pos, _ = forward_kinematics(quat_to_mat(q), r, sk["offsets"], sk["parents"])
    fwd = quat_to_mat(q[:, 0])[:, :, 2]; yaw = np.unwrap(np.arctan2(fwd[:, 0], fwd[:, 2]))
    hand = float((pos[:, ix["R_Wrist"], 1] - pos[:, ix["Head"], 1]).max())
    cms, worst = drift(pos, ix)
    print("%-60s seed %-3s %s  turned %6.0f°  went %.2f m  hips low %.2f  R hand over head %+.2f  feet drift %.1f cm/s planted (worst plant %.1f cm)  %.1fs" % (
        item["prompt"][:60], item.get("seed"), "loop" if item.get("loop") else "    ", np.degrees(yaw[-1] - yaw[0]),
        np.linalg.norm(r[-1, [0, 2]] - r[0, [0, 2]]), pos[:, 0, 1].min(), hand, cms, worst, time.time() - t0), flush=True)
