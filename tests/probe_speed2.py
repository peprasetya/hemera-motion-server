"""bf16 vs fp32 in joint positions after decoding; 50 vs 30 steps; with a few seeds."""
import os, sys, time
import numpy as np, torch
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(here))
import hymotion_engine as he
from motion_math import forward_kinematics
class NoText:
    def __init__(self, *a, **k): pass
he.TextEncoder = NoText
m = he.HYMotion(sys.argv[1], sys.argv[2], device="cuda")
p, dev = m.pipe, m.device
sk = m.skeleton
def sample(seed, dtype, steps):
    L = 120
    v = torch.cat([p.null_vtxt_feat.expand(1, 1, -1)] * 2)
    c = torch.cat([p.null_ctxt_input.expand(1, 128, -1)] * 2)
    cm = m.md.length_to_mask(torch.tensor([20, 20], device=dev), 128)
    xm = m.md.length_to_mask(torch.LongTensor([L, L]).to(dev), m.max_frames)
    y = p.noise_from_seeds(torch.zeros(1, m.max_frames, m.input_dim, device=dev), [seed], random_generator_on_gpu=False)
    t = torch.linspace(0, 1, steps + 1, device=dev)
    torch.cuda.synchronize(); t0 = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
        for i in range(steps):
            pred = p.motion_transformer(x=torch.cat([y] * 2), ctxt_input=c, vtxt_input=v, timesteps=t[i].expand(2), x_mask_temporal=xm, ctxt_mask_temporal=cm)
            u, cc = pred.float().chunk(2)
            y = y + (t[i + 1] - t[i]) * (u + 2.0 * (cc - u))
    torch.cuda.synchronize(); el = time.time() - t0
    with torch.no_grad():
        out = p.decode_motion_from_latent(y[:, :L], should_apply_smooothing=True)
        R = m.rot6d_to_matrix(out["rot6d"][0].to(dev)).double().cpu().numpy()
        root = (m.body.j_template[0] + out["transl"][0].to(dev)).double().cpu().numpy()
    pos, _ = forward_kinematics(R, root, sk["offsets"], sk["parents"])
    return pos - pos[:, :1, :] * np.array([1, 0, 1]), el
for seed in (1, 2, 3):
    ref, t32 = sample(seed, torch.float32, 50)
    b50, tb = sample(seed, torch.bfloat16, 50)
    b30, tb30 = sample(seed, torch.bfloat16, 30)
    f30, _ = sample(seed, torch.float32, 30)
    d = lambda a: np.linalg.norm(a - ref, axis=-1)
    print("seed %d: fp32 50 %.1fs | bf16 50 %.1fs: joints off by mean %.1f cm, max %.1f cm | bf16 30 %.1fs: mean %.1f max %.1f | fp32 30: mean %.1f max %.1f" % (
        seed, t32, tb, d(b50).mean() * 100, d(b50).max() * 100, tb30, d(b30).mean() * 100, d(b30).max() * 100, d(f30).mean() * 100, d(f30).max() * 100), flush=True)
