#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Two-stage training on vast.ai
#
# Usage:
#   bash scripts/run_cloud.sh <HF_TOKEN> <HF_REPO>
#
# Example:
#   bash scripts/run_cloud.sh hf_xxxx wh1te-wq/xbd-damage-segmentation
# ─────────────────────────────────────────────────────────────────────────────

set -e

HF_TOKEN="${1:?Usage: bash scripts/run_cloud.sh <HF_TOKEN> <HF_REPO>}"
HF_REPO="${2:?Usage: bash scripts/run_cloud.sh <HF_TOKEN> <HF_REPO>}"

# Paths relative to the repo root (where this script is run from)
ROOT="$(pwd)"
DATASET_DIR="${ROOT}/dataset"
WEIGHTS_S1="${ROOT}/weights/stage1"
WEIGHTS_S2="${ROOT}/weights/stage2"

echo "Root        : ${ROOT}"
echo "Dataset     : ${DATASET_DIR}"
echo "HF repo     : ${HF_REPO}"
echo ""

# Verify dataset exists
if [ ! -d "${DATASET_DIR}/train/post" ]; then
    echo "ERROR: dataset not found at ${DATASET_DIR}/train/post"
    echo "Make sure you extracted dataset.tar first:  tar -xf dataset.tar"
    exit 1
fi

echo "======================================================"
echo "  Stage 1: binary building segmentation"
echo "======================================================"
python -u scripts/train_stage1.py \
    --dataset_dir    "${DATASET_DIR}" \
    --checkpoint_dir "${WEIGHTS_S1}"

echo ""
echo "======================================================"
echo "  Stage 2: damage classification"
echo "======================================================"
python -u scripts/train_stage2.py \
    --dataset_dir    "${DATASET_DIR}" \
    --checkpoint_dir "${WEIGHTS_S2}"

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

upload_dir("${WEIGHTS_S1}", "weights/stage1")
upload_dir("${WEIGHTS_S2}", "weights/stage2")
print("Upload complete.")
PYEOF

echo ""
echo "======================================================"
echo "  All done. Shutting down instance in 30 seconds..."
echo "  (Press Ctrl-C to cancel)"
echo "======================================================"
sleep 30
sudo shutdown -h now
