# hemera-motion-server

Text to motion over HTTP. Words go in ("a slow twirl, arms out") and a few seconds of a person's
whole-body motion come out, on SMPL's 22-joint body skeleton, as JSON, binary glTF or BVH. It runs
[HY-Motion 1.0](https://github.com/Tencent-Hunyuan/HY-Motion-1.0) (Tencent, December 2025): a
flow-matching transformer with one billion parameters, with Qwen3-8B and CLIP-L as text encoders.

It is an engine for [hemera](https://github.com/peprasetya/hemera), the model router on deepmagpie. hemera starts it
when a motion is first asked for, passes requests and answers through unchanged, and stops it when
it has been idle a while. It also runs on its own. The API is documented with hemera's others, in
hemera's `docs/media-api.md` ("Motion"); in short:

| Call | Purpose |
|---|---|
| `POST /v1/motion/generations` | Start: `{"prompt", "duration" (1–10 s), "fps" (30), "seed", "loop"}`. Answers at once with an `id`. |
| `GET /v1/motion/generations/{id}` | `status` (`queued`, `in_progress`, `completed`, `failed`, `cancelled`), `progress`, `error`. |
| `GET /v1/motion/generations/{id}/content?format=json\|glb\|bvh` | The motion. |
| `DELETE /v1/motion/generations/{id}` | Cancel it or drop it. |
| `GET /health` | 200 once the model is loaded, 503 while it loads. |

**Private by design.**
- **Logging:** the prompt is never logged (its length is), and the access log is off.
- **Storage:** results live in memory only. They are gone after DELETE, after an hour, or when the
  server stops.
- **Network:** every model file is local and the Hugging Face libraries run offline, so nothing is
  fetched or sent.

## What is in here

| File | What it does |
|---|---|
| `motion_server.py` | The HTTP API, the queue (one clip at a time on the GPU), how long results are kept. |
| `hymotion_engine.py` | HY-Motion behind one call: loads Tencent's network unchanged, encodes the text, runs the 50 sampling steps, stands the body on the ground. |
| `motion_math.py` | Rotations, forward kinematics, loops, resampling to another frame rate. |
| `exports.py` | JSON, binary glTF (a skin with a stick-figure mesh and one animation) and BVH. |
| `engine.json` | What this engine serves, how to run it, what it downloads, and how its models look on disk. |
| `setup/fetch.sh`, `setup/models.tsv`, `setup/hy-motion-tree-c71ea6c.json` | The downloads, pinned and checked (below). |
| `setup/venv-from-comfyui.sh`, `setup/closure.py` | How the venv on deepmagpie was made. |
| `tests/` | `test_offline.py` (maths and formats, no GPU), `e2e.py` (through hemera, with Blender), `blender_check.py`, probes. |

## Setting it up

Everything goes in one directory, here `/media/shared/AIModels/motion`, which is called `DIR` below.

### 0. The server

```bash
git clone https://github.com/peprasetya/hemera-motion-server DIR/server
cd DIR/server
```

### 1. The downloads (about 22.3 GB)

```bash
setup/fetch.sh DIR
```

Each file is fetched at a fixed revision, resumably and one after another. Each is checked by
sha256 (large files) or by its git blob hash (small ones), and a file that fails is set aside as
`.bad`. Rerun the script to continue.

| What | From | Size | Licence |
|---|---|---|---|
| HY-Motion-1.0 weights: `latest.ckpt`, `config.yml` | huggingface.co/tencent/HY-Motion-1.0 @ `620dd559`, folder `HY-Motion-1.0` | 4.17 GB | Tencent HY-Motion 1.0 Community License |
| Qwen3-8B: 5 safetensors shards, tokenizer, config | huggingface.co/Qwen/Qwen3-8B @ `b968826d` | 16.4 GB | Apache-2.0 |
| CLIP ViT-L/14: `model.safetensors`, tokenizer, config | huggingface.co/openai/clip-vit-large-patch14 @ `32bd6428` | 1.72 GB | MIT |
| Tencent's code, and 10 Git LFS files it reads (body template, statistics) | github.com/Tencent-Hunyuan/HY-Motion-1.0 @ `c71ea6c` | ~6 MB | Tencent HY-Motion 1.0 Community License |

`setup/models.tsv` lists every file with its URL, size and hash.

Not needed, and not fetched:
- HY-Motion-1.0-Lite;
- Tencent's prompt rewriter and length estimator (Text2MotionPrompter, an LLM), because the
  client gives the duration;
- the 16 MB demo character (FBX).

**The licence, in short.**
- **Territory:** HY-Motion's licence does not apply in the European Union, the United Kingdom or
  South Korea.
- **Large services:** anyone whose products had more than 1 million monthly active users on the
  release date (2025-12-30) must ask Tencent for a licence.
- **Use of outputs:** they may not be used to improve other AI models. Tencent claims no rights in
  them.
- **Full text:** `DIR/models/HY-Motion-1.0/LICENSE.txt`.

### 2. Python

Python 3.12, PyTorch with a GPU, and the packages in `requirements.txt`.

**On deepmagpie** (AMD Radeon 8060S, gfx1151, ROCm 7.13), the venv was made without downloading
anything:

```bash
setup/venv-from-comfyui.sh DIR
```

It copies PyTorch 2.12 for ROCm 7.13 and the other packages from ComfyUI's venv, which it only
reads. That includes ComfyUI's ROCr library with the one-byte fix for ROCm 7.13's busy-waiting
event thread. Without the fix, a CPU core spins while the GPU works. ROCm 10 carries the upstream
fix.

**Any other machine:**

```bash
uv venv --python 3.12 DIR/venv
uv pip install --python DIR/venv/bin/python torch --index-url https://download.pytorch.org/whl/cu128   # or rocm6.4, or the right one for the GPU
uv pip install --python DIR/venv/bin/python -r requirements.txt
```

### 3. Try it on its own

```bash
cd /tmp && DIR/venv/bin/python -I motion_server.py --port 8190 --models DIR/models --code DIR/HY-Motion-1.0
curl -s localhost:8190/health
curl -s localhost:8190/v1/motion/generations -H 'Content-Type: application/json' \
     -d '{"prompt": "a person waves with the right hand", "duration": 4}'
curl -s localhost:8190/v1/motion/generations/<id>
curl -s 'localhost:8190/v1/motion/generations/<id>/content?format=glb' -o wave.glb
```

Run it from a directory other than this one, with `-I`. Tencent's code is imported from `--code`.

### 4. In hemera

```json
"engines": {
  "motion-server": {
    "binaryPath": "DIR/venv/bin/python",
    "argStyle": "custom",
    "healthPath": "/health",
    "env": { "HSA_TOOLS_DISABLE_REGISTER": "1", "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1" },
    "defaultArgs": ["-I", "DIR/server/motion_server.py", "--port", "{port}", "--models", "{file}"]
  }
},
"models": [
  { "name": "hy-motion-1.0", "engine": "motion-server", "task": "motion", "file": "DIR/models",
    "args": ["--code", "DIR/HY-Motion-1.0", "--name", "hy-motion-1.0"],
    "keepAliveSeconds": 600, "memoryMB": 24576 }
],
"motionModel": "hy-motion-1.0"
```

`"task": "motion"` is what makes hemera route `/v1/motion/generations` to it. `memoryMB` is how
much memory hemera makes room for before starting it.

The two environment settings matter on AMD GPUs:
- **`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL`:** turns on AMD's memory-efficient attention, which
  takes a clip from 9 s to 5.6 s.
- **`HSA_TOOLS_DISABLE_REGISTER`:** keeps ROCm 7.13 from spinning a core while idle.

The server sets both itself if they are missing. Elsewhere they do nothing.

## How it differs from running Tencent's code

Tencent's network, configuration, statistics, body template, decoding and smoothing are used
unchanged. What is ours:
- **Text encoders:** loaded with today's transformers. They give the network the same Qwen3-8B
  last hidden state and CLIP-L pooled output as Tencent's `HYTextModel`. Qwen3's language-model
  head (1.2 GB) is not loaded, since only its hidden states are used.
- **Sampling:** the same 50 Euler steps with classifier-free guidance (scale 5), written out so it
  reports progress and stops between steps when a clip is deleted.
- **Precision:** the network runs in bfloat16. On the Radeon 8060S that takes 5.6 s instead of
  29 s in float32, and moves the joints by 0.1–0.4 cm on average (1.6 cm at most) from the
  float32 result. `--precision fp32` gives Tencent's numbers. float16 overflows.
- **Checkpoint:** read with torch's safe loader, which runs no code from the file.
- **Ground:** found with the same skinning as Tencent's (checked to 3×10⁻⁷ m at every start), a
  few frames at a time.
- **Placement:** each clip is turned and moved so that its first frame stands at the origin facing
  +Z.
- **Loops** (`"loop": true`): HY-Motion has no loop mode, so the server makes one in three steps.
  1. It generates the clip a little longer, so there is a lead-in.
  2. It starts the loop where the pose and joint speeds best match the end, and spreads the
     drift (turn and distance over the ground) evenly over the clip.
  3. It blends the last frames into the frames that led up to the start, so the end runs into
     the start the way the motion ran into it.

  The last frame is then exactly the first.
- **Frame rate:** other frame rates are resampled from the model's 30 fps (slerp for rotations).

## Measured on deepmagpie

Measured on 2026-11-19 through hemera (`tests/e2e.py`, `tests/explore.py`, `tests/probe_more.py`).
The machine has a Ryzen AI Max+ 395, a Radeon 8060S (gfx1151) and 128 GB of memory shared with
the GPU, running ROCm 7.13.

| What | Measured |
|---|---|
| A clip of any length, 1 to 10 s | 5.8–6.3 s: the text 0.4 s, the 50 steps 5.3 s, the rest 0.1–0.2 s. The network always works on 12 s of frames, so the length does not change the time. |
| The first clip after a start | 43 s from disk, 34 s when the files are in the page cache. Most of it is loading 21 GB. |
| Memory while loaded | 20.5 GB on the GPU (`drm-total-gtt`) and 1.1 GB of ordinary memory. 20.7 GB on the GPU while making a clip. hemera makes room for 24 GB (`memoryMB`), which covers loading. |
| Output, 10 s at 60 fps | 600 frames: JSON 552 KB, GLB 261 KB, BVH 339 KB. |
| bfloat16 against float32 | Joints 0.1–0.4 cm apart on average, 1.6 cm at most. |

The four end-to-end motions, with fixed seeds:
- **"waves with the right hand":** the right hand is 3 cm over the head mid-wave. On three seeds
  it peaks 3–7 cm over the head, and the left hand never comes up.
- **"a slow twirl":** a 184° turn in 5 s.
- **"sits down":** the hips go from 0.99 m to 0.57 m.
- **An idle with `"loop": true`:** the last frame is the first, and the seam is as smooth as the
  rest of the clip.

Every GLB and BVH opens in Blender 5.3 as a 22-bone armature, posed within 0.03 mm of the JSON.

## What it does badly

- **Fingers** are not moved at all. The model makes the 22 body joints only, and the hands keep
  one relaxed shape. The skeleton has no finger joints.
- **Hands and the body.** The model knows nothing of the body's surface or of objects. A hand meant
  to rest on a thigh, a hip or the other hand comes close, but may hover or pass in. Sitting
  happens on a chair nobody sees (hips at 0.56 m).
- **Feet sliding.** Standing still is clean: planted feet move 0.3–0.4 cm a second, at most 2 cm.
  When the body travels or turns, planted feet slide:
  - 18 cm while stepping back to sit down;
  - up to 32 cm in one step while spinning;
  - 13–21 cm in an idle weight shift.

  Foot IK on the player's side fixes this where it shows.
- **Feet floating.** A clip stands on its single lowest point. If that is a heel or a toe at one
  moment, the feet can float 1–2 cm above the ground at other times.
- **How much.** Counts and amounts are followed loosely. "a slow twirl" gave a half turn on three
  seeds. "a person slowly spins around in a full circle" gave two to three turns, and not slowly.
- **Words.** Use English, and describe how the body moves, in at most about 60 words. Longer
  prompts are cut at 128 tokens (the job says `prompt_truncated`). Tencent passes other languages
  through a rewriting LLM, which is not used here. Appearance, emotions on their own, objects,
  scenery and other people are not understood.
- **Loops** are ours: a blend over the last fifth of the clip (6–24 frames). A motion that travels
  or turns a lot is pulled back to where it began, which shows as sliding. Loops suit idles.
- **No constraints:** keyframes, paths and target poses are not taken.

## Licence

This server's own code is MIT (see `LICENSE`). HY-Motion's code and weights are under Tencent's
licence above; Qwen3-8B is Apache-2.0; CLIP is MIT.
