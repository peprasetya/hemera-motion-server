"""HY-Motion 1.0 behind a small interface: words and a length in, at 30 frames a second, local
rotations of SMPL's 22 body joints and the root's path out.

Tencent's code (github.com/Tencent-Hunyuan/HY-Motion-1.0, at a fixed commit) is used unchanged
for the network, its configuration and statistics, the body template, and the decoding and
smoothing of what the network makes. What is ours:
- The text encoders are loaded with today's transformers. They give the network the same Qwen3-8B
  last hidden state (cropped to the user's text, 128 tokens) and CLIP-L pooled output that
  Tencent's HYTextModel computes; Qwen3's unused language-model head is not loaded.
- Sampling is the same 50-step Euler flow-matching loop with classifier-free guidance, written
  out so it can report progress and stop between steps. The network runs in bfloat16 (autocast):
  on the Radeon 8060S that is 5.6 s instead of 29 s in float32, and moves the joints by a mean of
  0.1-0.4 cm (at most 1.6 cm) from the float32 result. --precision fp32 gives Tencent's numbers.
- The checkpoint is read with torch's safe loader, which runs no code from the file.
- The ground is found with the same skinning as Tencent's, done a few frames at a time.
"""
import json
import os
import sys
import time
import types

import numpy as np
import torch
import yaml

from exports import Clip
from motion_math import continuous, loop_plan, make_loop, mat_to_quat, resample, start_at_origin

BODY_JOINTS = 22


class Cancelled(Exception):
    pass


class TextEncoder:
    def __init__(self, qwen_dir, clip_dir, system, device, max_len):
        from transformers import AutoModel, AutoTokenizer, CLIPTextModel, CLIPTokenizer
        self.device = device
        self.system = system
        self.max_len = max_len
        self.tok = AutoTokenizer.from_pretrained(qwen_dir, padding_side="right")
        self.llm = AutoModel.from_pretrained(qwen_dir, dtype=torch.bfloat16).eval().requires_grad_(False).to(device)
        self.ctok = CLIPTokenizer.from_pretrained(clip_dir, max_length=77)
        self.clip = CLIPTextModel.from_pretrained(clip_dir).eval().requires_grad_(False).to(device)
        self.crop = self._crop_start()

    def _render(self, text):
        msgs = [{"role": "system", "content": self.system}, {"role": "user", "content": text}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False)

    def _crop_start(self):
        full = self.tok(self._render("<BOC>"), add_special_tokens=True)["input_ids"]
        marker = self.tok("<BOC>", add_special_tokens=False)["input_ids"]
        for i in range(len(full) - len(marker) + 1):
            if full[i:i + len(marker)] == marker:
                return i
        return max(0, len(full) - 1)

    @torch.no_grad()
    def encode(self, text):
        rendered = self._render(text)
        truncated = len(self.tok(rendered)["input_ids"]) > self.max_len + self.crop
        enc = self.tok([rendered], truncation=True, max_length=self.max_len + self.crop, padding="max_length",
                       return_tensors="pt", return_attention_mask=True)
        ids, mask = enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)
        hidden = self.llm(input_ids=ids, attention_mask=mask).last_hidden_state
        ctxt = hidden[:, self.crop:self.crop + self.max_len].contiguous()
        length = (mask.sum(dim=-1) - self.crop).clamp(min=0, max=self.max_len)
        c = self.ctok([text], truncation=True, max_length=77, padding=True, return_tensors="pt",
                      return_attention_mask=True)
        vtxt = self.clip(input_ids=c["input_ids"].to(self.device),
                         attention_mask=c["attention_mask"].to(self.device)).pooler_output.unsqueeze(1)
        return vtxt, ctxt, length, truncated


class HYMotion:
    fps = 30

    def __init__(self, code_dir, models_dir, device="cuda", log=print, precision="bf16"):
        self.log = log
        self.device = torch.device(device)
        self.autocast = {"bf16": torch.bfloat16, "fp32": None}[precision]
        code_dir = os.path.abspath(code_dir)
        models_dir = os.path.abspath(models_dir)
        os.chdir(code_dir)  # Tencent's code finds its body template and statistics from here
        sys.path.insert(0, code_dir)
        shim = types.ModuleType("torchdiffeq")  # imported by motion_diffusion; sampling is done here

        def odeint(*_a, **_k):
            raise RuntimeError("sampling runs in hymotion_engine")

        shim.odeint = odeint
        sys.modules.setdefault("torchdiffeq", shim)
        from hymotion.network.text_encoders.model_constants import PROMPT_TEMPLATE_ENCODE_HUMAN_MOTION
        from hymotion.pipeline import motion_diffusion
        from hymotion.utils.geometry import rot6d_to_rotation_matrix
        from hymotion.utils.loaders import load_object
        self.md = motion_diffusion
        self.rot6d_to_matrix = rot6d_to_rotation_matrix

        t0 = time.time()
        with open(os.path.join(models_dir, "HY-Motion-1.0", "config.yml")) as f:
            cfg = yaml.safe_load(f)
        pipe = load_object(cfg["train_pipeline"], cfg["train_pipeline_args"], network_module=cfg["network_module"],
                           network_module_args=cfg["network_module_args"])
        ckpt = torch.load(os.path.join(models_dir, "HY-Motion-1.0", "latest.ckpt"), map_location="cpu",
                          weights_only=True, mmap=True)
        state = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = pipe.load_state_dict(state, strict=False)
        lost = [k for k in missing if k.startswith("motion_transformer.") or k.startswith("null_")]
        if lost:
            raise RuntimeError("the checkpoint lacks %d network weights, e.g. %s" % (len(lost), lost[0]))
        del ckpt, state
        pipe.eval().requires_grad_(False)
        self.body = pipe.body_model  # Tencent's wooden body: rest joints, skin, parents
        pipe.body_model = None  # decode without its whole-mesh pass; the ground is found below
        pipe.to(self.device)
        self.body.to(self.device)
        self.pipe = pipe
        self.max_frames = int(pipe.train_frames)
        self.input_dim = int(cfg["network_module_args"]["input_dim"])
        self.log("motion network loaded in %.1f s (missing %d, unexpected %d keys: %s)" % (
            time.time() - t0, len(missing), len(unexpected), sorted({k.split(".")[0] for k in missing}) or "-"))

        t0 = time.time()
        self.text = TextEncoder(os.path.join(models_dir, "Qwen3-8B"), os.path.join(models_dir, "clip-vit-large-patch14"),
                                PROMPT_TEMPLATE_ENCODE_HUMAN_MOTION, self.device,
                                int(cfg["train_pipeline_args"]["text_encoder_cfg"]["max_length_llm"]))
        self.log("text encoders loaded in %.1f s" % (time.time() - t0))
        self.skeleton = self._skeleton()
        self.log("skinning check: %.2e m" % self._check_skinning())

    # ---- the skeleton clients get ------------------------------------------------------------------

    def _skeleton(self):
        b = self.body
        j = b.j_template.double().cpu().numpy()
        v = b.v_template.double().cpu().numpy()
        names = list(b.joint_names)
        parents = [int(p) for p in b.parents.cpu().numpy()[:BODY_JOINTS]]
        parents[0] = -1
        ground = v[:, 1].min()
        offsets = np.array([j[i] - j[parents[i]] if i else j[0] - [0.0, ground, 0.0] for i in range(BODY_JOINTS)])
        ix = {n: i for i, n in enumerate(names)}

        def toe(foot):
            near = (np.abs(v[:, 0] - j[foot, 0]) < 0.06) & (v[:, 1] < j[foot, 1] + 0.05)
            return [0.0, 0.0, float(v[near, 2].max() - j[foot, 2])] if near.any() else [0.0, 0.0, 0.05]

        ends = {
            ix["Head"]: [0.0, float(v[:, 1].max() - j[ix["Head"], 1]), 0.0],
            ix["L_Wrist"]: (j[ix["L_Middle1"]] - j[ix["L_Wrist"]]).tolist(),
            ix["R_Wrist"]: (j[ix["R_Middle1"]] - j[ix["R_Wrist"]]).tolist(),
            ix["L_Foot"]: toe(ix["L_Foot"]),
            ix["R_Foot"]: toe(ix["R_Foot"]),
        }
        return {"joints": names[:BODY_JOINTS], "parents": parents, "offsets": offsets.tolist(), "ends": ends}

    # ---- skinning, to stand the clip on the ground -------------------------------------------------

    def _skinning(self, R):
        """World transforms of the 52-joint body for local rotations R (L, 22, 3, 3); the fingers
        stay in their rest pose, as in Tencent's WoodenMesh."""
        b = self.body
        J = b.j_template.shape[0]
        L = R.shape[0]
        R = torch.cat([R, torch.eye(3, device=R.device, dtype=R.dtype).expand(L, J - R.shape[1], 3, 3)], dim=1)
        parents = b.parents.tolist()
        j = b.j_template.to(R.dtype)
        rel = j.clone()
        rel[1:] = j[1:] - j[parents[1:]]
        gr, gt = [R[:, 0]], [rel[0].expand(L, 3)]
        for i in range(1, J):
            p = parents[i]
            gr.append(gr[p] @ R[:, i])
            gt.append(gt[p] + torch.einsum("lab,b->la", gr[p], rel[i]))
        gr, gt = torch.stack(gr, 1), torch.stack(gt, 1)
        return gr, gt - torch.einsum("ljab,jb->lja", gr, j)

    def _vertices(self, gr, at):
        b = self.body
        v, w, idx = b.v_template.to(gr.dtype), b.skin_weights.to(gr.dtype), b.skin_indices
        out = 0
        for k in range(idx.shape[1]):
            out = out + w[:, k, None] * (torch.einsum("cvab,vb->cva", gr[:, idx[:, k]], v) + at[:, idx[:, k]])
        return out

    def _lowest_point(self, R, transl, chunk=32):
        gr, at = self._skinning(R)
        low = None
        for s in range(0, R.shape[0], chunk):
            y = self._vertices(gr[s:s + chunk], at[s:s + chunk])[..., 1] + transl[s:s + chunk, 1, None]
            m = float(y.min())
            low = m if low is None else min(low, m)
        return low

    @torch.no_grad()
    def _check_skinning(self):
        """Our chunked skinning against Tencent's WoodenMesh, on a few random poses."""
        g = torch.Generator().manual_seed(0)
        rot6d = torch.randn(4, BODY_JOINTS, 6, generator=g).to(self.device)
        R = self.rot6d_to_matrix(rot6d)
        theirs = self.body.forward({"rot6d": rot6d, "trans": torch.zeros(4, 3, device=self.device)})["vertices"]
        gr, at = self._skinning(R)
        return float((self._vertices(gr, at) - theirs).abs().max())

    # ---- making a clip -----------------------------------------------------------------------------

    @torch.no_grad()
    def _sample(self, prompt, frames, seed, progress, cancelled, timing):
        p, md = self.pipe, self.md
        t0 = time.time()
        vtxt, ctxt, clen, truncated = self.text.encode(prompt)
        timing["text"] = int((time.time() - t0) * 1000)
        t0 = time.time()
        ctxt = ctxt.float()
        ctxt_mask = md.length_to_mask(clen, ctxt.shape[1])
        x_mask = md.length_to_mask(torch.LongTensor([frames]).to(self.device), self.max_frames)
        null_c = p.null_ctxt_input.expand(*ctxt.shape) if p.enable_ctxt_null_feat else ctxt
        v_in = torch.cat([p.null_vtxt_feat.expand(*vtxt.shape), vtxt], dim=0)
        c_in = torch.cat([null_c, ctxt], dim=0)
        x_m, c_m = torch.cat([x_mask] * 2, dim=0), torch.cat([ctxt_mask] * 2, dim=0)
        y = p.noise_from_seeds(torch.zeros(1, self.max_frames, self.input_dim, device=self.device), [seed],
                               random_generator_on_gpu=p.random_generator_on_gpu)
        steps = int(p.validation_steps)
        t = torch.linspace(0, 1, steps + 1, device=self.device)
        scale = float(p.text_guidance_scale)
        for i in range(steps):
            if cancelled():
                raise Cancelled()
            with torch.autocast(self.device.type, dtype=self.autocast or torch.float32, enabled=self.autocast is not None):
                pred = p.motion_transformer(x=torch.cat([y] * 2, dim=0), ctxt_input=c_in, vtxt_input=v_in,
                                            timesteps=t[i].expand(2), x_mask_temporal=x_m, ctxt_mask_temporal=c_m)
            uncond, cond = pred.float().chunk(2, dim=0)
            y = y + (t[i + 1] - t[i]) * (uncond + scale * (cond - uncond))
            progress(i + 1, steps)
        timing["sample"] = int((time.time() - t0) * 1000)
        t0 = time.time()
        out = p.decode_motion_from_latent(y[:, :frames], should_apply_smooothing=True)
        R = self.rot6d_to_matrix(out["rot6d"][0, :, :BODY_JOINTS].to(self.device))
        transl = out["transl"][0].to(self.device)
        transl[:, 1] -= self._lowest_point(R, transl)  # stand on y = 0, as Tencent's decoder does
        root = self.body.j_template[0] + transl
        timing["decode"] = int((time.time() - t0) * 1000)
        return R.double().cpu().numpy(), root.double().cpu().numpy(), truncated

    def make(self, prompt, duration, fps, loop, seed, model_name, progress=lambda i, n: None,
             cancelled=lambda: False):
        timing = {}
        n30 = int(round(duration * self.fps))
        total, blend = loop_plan(n30, self.max_frames) if loop else (n30, 0)
        R, root, truncated = self._sample(prompt, total, seed, progress, cancelled, timing)
        sk = self.skeleton
        if loop:
            R, root, _ = make_loop(R, root, n30, blend, sk["offsets"], sk["parents"])
        R, root = start_at_origin(R, root)
        q, root = resample(continuous(mat_to_quat(R)), root, int(round(duration * fps)))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return Clip(model_name, sk, q, root, fps, loop, seed), timing, truncated

    def describe(self):
        return json.dumps(self.skeleton)
