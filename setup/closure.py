# Run with ComfyUI's venv python (read only): print the site-packages entries of some
# distributions and of everything they need (no extras), to copy into a venv of the same Python.
import importlib.metadata as md, sys
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name as canon

skip = {canon(x) for x in sys.argv[2].split(",")} if len(sys.argv) > 2 else set()
todo, seen = [canon(x) for x in sys.argv[1].split(",")], set()
env = {"extra": ""}
while todo:
    n = todo.pop()
    if n in seen or n in skip:
        continue
    seen.add(n)
    d = md.distribution(n)
    for r in d.requires or []:
        req = Requirement(r)
        if req.marker and not req.marker.evaluate(env):
            continue
        todo.append(canon(req.name))
tops = set()
for n in sorted(seen):
    d = md.distribution(n)
    for f in d.files or []:
        top = str(f).split("/")[0]
        if top.endswith(".pth") or top == ".." or top.startswith("__pycache__"):
            continue
        tops.add(top)
print("\n".join(sorted(tops)))
print("DISTS " + " ".join(sorted(f"{n}=={md.distribution(n).version}" for n in seen)), file=sys.stderr)
