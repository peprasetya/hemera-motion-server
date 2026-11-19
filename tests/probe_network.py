"""Load only HY-Motion's network and body template (no text encoders): checkpoint keys, skeleton,
our skinning against Tencent's, and one unconditioned sample, to try the GPU path early."""
import os
import sys
import time

import numpy as np
import torch

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(here))
import hymotion_engine as he  # noqa: E402

code, models = sys.argv[1], sys.argv[2]


class NoText:
    def __init__(self, *a, **k):
        pass


he.TextEncoder = NoText
t0 = time.time()
m = he.HYMotion(code, models, device="cuda")
print("loaded in %.1f s" % (time.time() - t0))
ck = torch.load(os.path.join(models, "HY-Motion-1.0", "latest.ckpt"), map_location="cpu", weights_only=True, mmap=True)
print("checkpoint top-level keys:", list(ck.keys()))
sk = m.skeleton
print("joints:", sk["joints"])
print("parents:", sk["parents"])
for n, o in zip(sk["joints"], sk["offsets"]):
    print("  %-11s %s" % (n, np.round(o, 3)))
print("ends:", {sk["joints"][k]: np.round(v, 3).tolist() for k, v in sk["ends"].items()})

# unconditioned sample through the network (null text features), 120 frames
p = m.pipe
dev = m.device
L = 120
with torch.no_grad():
    v = p.null_vtxt_feat.expand(1, 1, -1)
    c = p.null_ctxt_input.expand(1, 1, -1)
    cl = torch.tensor([1], device=dev)
    cm = m.md.length_to_mask(cl, 1)
    xm = m.md.length_to_mask(torch.LongTensor([L]).to(dev), m.max_frames)
    y = p.noise_from_seeds(torch.zeros(1, m.max_frames, m.input_dim, device=dev), [7], random_generator_on_gpu=False)
    t = torch.linspace(0, 1, 51, device=dev)
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(50):
        pred = p.motion_transformer(x=y, ctxt_input=c, vtxt_input=v, timesteps=t[i].expand(1), x_mask_temporal=xm,
                                    ctxt_mask_temporal=cm)
        y = y + (t[i + 1] - t[i]) * pred
    torch.cuda.synchronize()
    print("50 steps, batch 1, fp32: %.2f s" % (time.time() - t0))
    out = p.decode_motion_from_latent(y[:, :L], should_apply_smooothing=True)
    R = m.rot6d_to_matrix(out["rot6d"][0].to(dev))
    tr = out["transl"][0].to(dev)
    print("lowest point before standing it up: %.3f m" % m._lowest_point(R, tr))
print("allocated %.2f GB, reserved %.2f GB" % (torch.cuda.memory_allocated() / 1e9, torch.cuda.memory_reserved() / 1e9))
