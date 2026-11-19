#!/bin/bash
# Fetch what the motion server needs into DIR, one file after another (about 22.3 GB):
#   DIR/HY-Motion-1.0/   Tencent's code at a fixed commit, with the Git LFS files it reads
#   DIR/models/          HY-Motion-1.0 weights, Qwen3-8B, clip-vit-large-patch14 (see models.tsv)
# Resumable: rerun it and it continues (.part files), skipping files already verified.
# Every file is checked: LFS files by sha256, small files by their git blob sha1.
#
#   setup/fetch.sh [DIR]        (default /media/shared/AIModels/motion)
set -u
BASE=${1:-/media/shared/AIModels/motion}
HERE=$(cd "$(dirname "$0")" && pwd)
REV=c71ea6c694d8f221c742ff435d9447fa88f91707

verify() { # file kind hash size
  case "$2" in
    sha256) [ "$(sha256sum "$1" | cut -d' ' -f1)" = "$3" ] ;;
    gitsha1) [ "$( (printf 'blob %s\0' "$4"; cat "$1") | sha1sum | cut -d' ' -f1)" = "$3" ] ;;
  esac
}

fetch() { # url dest size kind hash
  local url=$1 dest=$2 size=$3 kind=$4 hash=$5
  mkdir -p "$(dirname "$dest")"
  if [ -f "$dest" ] && [ "$(stat -c %s "$dest")" = "$size" ] && verify "$dest" "$kind" "$hash" "$size"; then
    echo "have  $dest"; return 0
  fi
  for try in 1 2 3 4 5 6 7 8; do
    have=$(stat -c %s "$dest.part" 2>/dev/null || echo 0)
    [ "$have" -gt "$size" ] && rm -f "$dest.part"
    [ "$have" = "$size" ] && break
    echo "$(date +%T) get   $dest (try $try, have $have of $size)"
    curl -fsSL --retry 5 --retry-delay 10 --connect-timeout 30 -C - -o "$dest.part" "$url" && break
    sleep 15
  done
  if [ "$(stat -c %s "$dest.part" 2>/dev/null || echo 0)" != "$size" ]; then echo "FAIL  size $dest"; return 1; fi
  if ! verify "$dest.part" "$kind" "$hash" "$size"; then echo "FAIL  hash $dest"; mv "$dest.part" "$dest.bad"; return 1; fi
  mv "$dest.part" "$dest"; echo "$(date +%T) ok    $dest"
}

fail=0
# 1. Tencent's code at a fixed commit, and the Git LFS files the code reads.
mkdir -p "$BASE/src"
tgz="$BASE/src/HY-Motion-1.0-$REV.tar.gz"
if [ ! -f "$tgz" ]; then
  curl -fsSL --retry 5 -o "$tgz.part" "https://codeload.github.com/Tencent-Hunyuan/HY-Motion-1.0/tar.gz/$REV" && mv "$tgz.part" "$tgz"
fi
if [ ! -d "$BASE/HY-Motion-1.0" ]; then
  mkdir -p "$BASE/HY-Motion-1.0" && tar -xzf "$tgz" -C "$BASE/HY-Motion-1.0" --strip-components=1 || fail=1
fi
python3 -I - "$BASE/HY-Motion-1.0" "$HERE/hy-motion-tree-c71ea6c.json" <<'PY' || fail=1
import hashlib, json, os, sys
root, tree = sys.argv[1], json.load(open(sys.argv[2]))
bad = 0
for t in tree["tree"]:
    if t["type"] != "blob": continue
    p = os.path.join(root, t["path"])
    if os.path.exists(p + ".lfs-pointer"): p += ".lfs-pointer"  # the real file replaced it
    data = open(p, "rb").read() if os.path.exists(p) else None
    if data is None or hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest() != t["sha"]:
        print("FAIL  code", t["path"]); bad += 1
print("code  %d files checked against commit, %d bad" % (len(tree["tree"]), bad))
sys.exit(1 if bad else 0)
PY
for f in stats/Mean.npy stats/Std.npy scripts/gradio/static/assets/dump_wooden/{faces,j_template,keypoints,kintree,skinIndice,skinWeights,uvs,v_template}.bin; do
  p="$BASE/HY-Motion-1.0/$f"
  [ -f "$p.lfs-pointer" ] || cp "$p" "$p.lfs-pointer"   # as extracted, the file is only the pointer
  oid=$(grep -a "^oid sha256:" "$p.lfs-pointer" | cut -d: -f2); sz=$(grep -a "^size " "$p.lfs-pointer" | cut -d' ' -f2)
  fetch "https://media.githubusercontent.com/media/Tencent-Hunyuan/HY-Motion-1.0/$REV/$f" "$p" "$sz" sha256 "$oid" || fail=1
done

# 2. The weights, smallest first.
while IFS=$'\t' read -r url out size kind hash; do
  [ -z "$url" ] && continue
  fetch "$url" "$BASE/$out" "$size" "$kind" "$hash" || fail=1
done < "$HERE/models.tsv"

df -h "$BASE" | tail -1
[ $fail = 0 ] && echo "ALL DONE" || echo "SOME FAILED (rerun to resume)"
