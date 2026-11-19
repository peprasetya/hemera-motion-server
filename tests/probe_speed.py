"""Warm speed of HY-Motion's network: fp32 vs bf16 autocast, batch 2 (as with guidance), and how
far bf16 moves the result."""
import os, sys, time, warnings
import torch
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(here))
import hymotion_engine as he
class NoText:
    def __init__(self, *a, **k): pass
he.TextEncoder = NoText
warnings.simplefilter("once")
m = he.HYMotion(sys.argv[1], sys.argv[2], device="cuda")
p, dev = m.pipe, m.device
def run(L, dtype, steps=50):
    v = torch.cat([p.null_vtxt_feat.expand(1, 1, -1)] * 2)
    c = torch.cat([p.null_ctxt_input.expand(1, 128, -1)] * 2)
    cm = m.md.length_to_mask(torch.tensor([20, 20], device=dev), 128)
    xm = m.md.length_to_mask(torch.LongTensor([L, L]).to(dev), m.max_frames)
    y = p.noise_from_seeds(torch.zeros(1, m.max_frames, m.input_dim, device=dev), [7], random_generator_on_gpu=False)
    t = torch.linspace(0, 1, steps + 1, device=dev)
    torch.cuda.synchronize(); t0 = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
        for i in range(steps):
            pred = p.motion_transformer(x=torch.cat([y] * 2), ctxt_input=c, vtxt_input=v, timesteps=t[i].expand(2),
                                        x_mask_temporal=xm, ctxt_mask_temporal=cm)
            u, cc = pred.float().chunk(2)
            y = y + (t[i + 1] - t[i]) * (u + 5.0 * (cc - u))
    torch.cuda.synchronize()
    return time.time() - t0, y[:, :L]
for dtype in (torch.float32, torch.bfloat16, torch.float16):
    for L in (120, 300):
        run(L, dtype, steps=3)  # warm
        s, y = run(L, dtype)
        print("%-15s %3d frames: %.2f s for 50 steps (%.0f ms/step)" % (str(dtype), L, s, s * 20), flush=True)
        if dtype == torch.float32 and L == 120: ref = y
        if L == 120 and dtype != torch.float32:
            print("   max |latent - fp32| = %.4f (latent std 1)" % float((y - ref).abs().max()))
