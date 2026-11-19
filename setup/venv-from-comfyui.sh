#!/bin/bash
# How the venv on deepmagpie was made, with nothing downloaded: PyTorch for ROCm 7.13 (gfx1151,
# Radeon 8060S) and the other packages copied from ComfyUI's venv, which has the same Python 3.12.
# ComfyUI's venv is only read. Elsewhere, use the README's "Any other machine" steps instead.
#
#   setup/venv-from-comfyui.sh [DIR] [COMFYUI_VENV]
set -eu
DIR=${1:-/media/shared/AIModels/motion}
C=${2:-/media/shared/AIModels/ComfyUI/comfyui/venv}
HERE=$(cd "$(dirname "$0")" && pwd)
PY=$(readlink -f "$C/bin/python")
SP_C=$(echo "$C"/lib/python3.*/site-packages)
[ -d "$DIR/venv" ] || ~/.local/bin/uv venv --python "$PY" "$DIR/venv"
SP=$(echo "$DIR"/venv/lib/python3.*/site-packages)

# 1. PyTorch and ROCm (triton is left out: it is only for torch.compile, which is not used)
cd "$SP_C"
for d in torch torchgen functorch torch-*.dist-info _rocm_sdk_core _rocm_sdk_libraries_* rocm_sdk rocm_sdk_core \
         rocm_sdk_libraries_* rocm-*.dist-info rocm_sdk_core-*.dist-info rocm_sdk_libraries_*-*.dist-info; do
  [ -e "$SP/$d" ] || cp -a "$d" "$SP/"
done
# ComfyUI's ROCr library carries the one-byte fix for ROCm 7.13's busy-waiting event thread
# (upstream fix c06ea68a592f, in ROCm 10); its unfixed original is not needed here.
rm -f "$SP"/_rocm_sdk_core/lib/libhsa-runtime64.so.1.orig-*

# 2. Everything else the server imports, with what those need
"$C/bin/python" -I "$HERE/closure.py" numpy,scipy,pyyaml,einops,safetensors,transformers,filelock,typing-extensions,sympy,networkx,jinja2,fsspec,setuptools torch,triton,rocm > "$DIR/venv/copied.txt"
while read -r e; do [ -e "$SP/$e" ] || cp -a "$e" "$SP/"; done < "$DIR/venv/copied.txt"

cd /tmp && HSA_TOOLS_DISABLE_REGISTER=1 "$DIR/venv/bin/python" -I -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__, 'gpu', torch.cuda.is_available())"
