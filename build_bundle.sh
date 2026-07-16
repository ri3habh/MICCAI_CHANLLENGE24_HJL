#!/usr/bin/env bash
#
# build_bundle.sh — unpack weights -> build the GPU Docker image -> test it.
# Optionally also export a shippable tarball (EXPORT_TARBALL=1).
#
# Typical use (build on the GPU machine, then run it there — see HANDOFF.md):
#     WEIGHTS_ZIP=/path/to/weights.zip ./build_bundle.sh
#
# Requires: Docker installed + network access to pull the CUDA PyTorch base
# image (~5 GB) the first time. A GPU is NOT needed to BUILD — only to run later.
#
# EXPORT_TARBALL=1 additionally does `docker save | gzip` to ship a pre-built
# image to a machine that can't build (needs ~8-12 GB free for the tarball).
set -euo pipefail

# --- config (override by exporting these before running) --------------------
REPO_ROOT="$(cd "$(dirname "$0")" && pwd -P)"
ZIP="${WEIGHTS_ZIP:-/c/Users/Rishabh/Downloads/批量下载-resources等45个文件.zip}"
IMAGE="${IMAGE_TAG:-aortaseg-validate}"
OUT_TARBALL="${OUT_TARBALL:-$REPO_ROOT/aortaseg-validate.tar.gz}"
RUN_INIMAGE_TESTS="${RUN_INIMAGE_TESTS:-1}"   # set to 0 to skip the sanity test
EXPORT_TARBALL="${EXPORT_TARBALL:-0}"         # set to 1 to also save a shippable tarball

RESENCL="$REPO_ROOT/resources/Dataset040_Aortaseg24/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/dataset.json"

cd "$REPO_ROOT"

# --- 1. unpack the weights into ./resources (idempotent) --------------------
if [ -f "$RESENCL" ]; then
    echo "[1/3] weights already present at resources/ — skipping extraction."
else
    echo "[1/3] extracting weights from: $ZIP"
    [ -f "$ZIP" ] || { echo "ERROR: weights zip not found: $ZIP"; echo "Set WEIGHTS_ZIP=/path/to/zip and re-run."; exit 1; }
    unzip -q -o "$ZIP" -d "$REPO_ROOT"    # zip is rooted at resources/, lands as ./resources/...
fi

# verify every checkpoint the ensemble loads is actually there
echo "      verifying model folds..."
missing=0
for f in \
    "nnUNetTrainer__nnUNetPlans__3d_lowres/fold_0/checkpoint_final.pth" \
    "nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/fold_0/checkpoint_final.pth" \
    "nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/fold_1/checkpoint_final.pth" \
    "nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/fold_2/checkpoint_final.pth" \
    "nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/fold_4/checkpoint_final.pth" \
    "nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/fold_all/checkpoint_final.pth" ; do
    if [ ! -f "$REPO_ROOT/resources/Dataset040_Aortaseg24/$f" ]; then
        echo "      MISSING: resources/Dataset040_Aortaseg24/$f"; missing=1
    fi
done
[ "$missing" -eq 0 ] || { echo "ERROR: weights incomplete — aborting."; exit 1; }
echo "      OK: all 6 ensemble checkpoints present."

# --- 2. build the GPU image (weights + code baked in) -----------------------
echo "[2] docker build -f Dockerfile.validation -t $IMAGE ."
docker build -f Dockerfile.validation -t "$IMAGE" "$REPO_ROOT"

if [ "$RUN_INIMAGE_TESTS" = "1" ]; then
    echo "      running unit tests inside the image (no GPU needed)..."
    docker run --rm --entrypoint python "$IMAGE" -m pytest -q
fi

# --- 3. (optional) export a shippable tarball -------------------------------
if [ "$EXPORT_TARBALL" = "1" ]; then
    echo "[3] docker save $IMAGE -> $OUT_TARBALL (this takes a while)"
    docker save "$IMAGE" | gzip > "$OUT_TARBALL"
    echo "      wrote $OUT_TARBALL ($(ls -lh "$OUT_TARBALL" | awk '{print $5}'))"
fi

echo
echo "DONE. Image '$IMAGE' is built and its unit tests pass."
echo "Next: run preflight, then the batch (see HANDOFF.md, section 3-4)."
echo "Smoke-test 2 cases first with --limit 2 before the full run."
