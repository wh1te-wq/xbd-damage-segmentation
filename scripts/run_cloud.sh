#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Two-stage training on vast.ai
#
# Usage (paste into vast.ai terminal after setup):
#   bash scripts/run_cloud.sh <HF_TOKEN> <HF_REPO>
#
# Example:
#   bash scripts/run_cloud.sh hf_xxxx wh1te-wq/xbd-damage-segmentation
#
# What this does:
#   1. Train Stage 1 (binary building segmentation, 40 epochs)
#   2. Train Stage 2 (damage classification, 60 epochs)
#   3. Upload weights + logs to HuggingFace
#   4. Shut down the instance  (stops billing automatically)
# ─────────────────────────────────────────────────────────────────────────────

set -e   # stop immediately if any command fails

HF_TOKEN="${1:?Usage: bash run_cloud.sh <HF_TOKEN> <HF_REPO>}"
HF_REPO="${2:?Usage: bash run_cloud.sh <HF_TOKEN> <HF_REPO>}"

echo "======================================================"
echo "  Stage 1: binary building segmentation"
echo "======================================================"
python -u scripts/train_stage1.py

echo ""
echo "======================================================"
echo "  Stage 2: damage classification"
echo "======================================================"
python -u scripts/train_stage2.py

echo ""
echo "======================================================"
echo "  Uploading results to HuggingFace: ${HF_REPO}"
echo "======================================================"
python -u - <<PYEOF
from huggingface_hub import HfApi
import os, glob

api   = HfApi()
token = "${HF_TOKEN}"
repo  = "${HF_REPO}"

def upload_dir(local_dir, remote_dir):
    files = glob.glob(os.path.join(local_dir, "*"))
    if not files:
        print(f"  (no files in {local_dir}, skipping)")
        return
    for path in files:
        fname = os.path.basename(path)
        dest  = f"{remote_dir}/{fname}"
        print(f"  uploading {path} -> {dest}")
        api.upload_file(
            path_or_fileobj = path,
            path_in_repo    = dest,
            repo_id         = repo,
            token           = token,
            repo_type       = "model",
        )

upload_dir("weights/stage1", "weights/stage1")
upload_dir("weights/stage2", "weights/stage2")
print("Upload complete.")
PYEOF

echo ""
echo "======================================================"
echo "  All done. Shutting down instance in 30 seconds..."
echo "  (Press Ctrl-C to cancel shutdown)"
echo "======================================================"
sleep 30
sudo shutdown -h now
