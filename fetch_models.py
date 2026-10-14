"""Download the winter checkpoints from Hugging Face into results/<tag>.pt,
the format t7_pas.load() reads.

  python fetch_models.py                  all 45 models
  python fetch_models.py gram_pas1        only tags containing "gram_pas1"
  python fetch_models.py --local DIR      convert a local copy of the Hugging Face repo instead
"""

import os
import sys

import torch
from safetensors.torch import load_file

REPO_ID = "99kj/where-winter-goes"


def main(args):
    local = None
    if args[:1] == ["--local"]:
        local, args = args[1], args[2:]
    match = args[0] if args else ""
    if local is None:
        from huggingface_hub import snapshot_download
        local = snapshot_download(REPO_ID, allow_patterns=[f"models/*{match}*.safetensors"])
    src = os.path.join(local, "models")
    os.makedirs("results", exist_ok=True)
    n = 0
    for fn in sorted(os.listdir(src)):
        if fn.endswith(".safetensors") and match in fn:
            tag = fn[: -len(".safetensors")]
            torch.save(load_file(os.path.join(src, fn)), os.path.join("results", f"{tag}.pt"))
            n += 1
    print(f"wrote {n} checkpoints to results/")


if __name__ == "__main__":
    main(sys.argv[1:])
