#!/usr/bin/env bash
#
# build_bundle.sh — one-shot: unpack weights -> build the GPU Docker image ->
# (optionally) test it -> export the shippable tarball for the coworker's 4090.
#
# Run from the repo root in Git Bash:
#     ./build_bundle.sh
#
# Requires: Docker installed (a GPU is NOT needed to BUILD — only to run later).
# The build downloads the CUDA PyTorch base image (~5 GB) the first time, and the
# final tarball is large (weights are baked in), so expect ~8-12 GB out.
set -euo pipefail

# --- config (override by exporting these before running) --------------------
REPO_ROOT="$(cd "$(dirname "$0")" && pwd -P)"
ZIP="${WEIGHTS_ZIP:-/c/Users/Rishabh/Downloads/批量下载-resources等45个文件.zip}"
IMAGE="${IMAGE_TAG:-aortaseg-validate}"
OUT_TARBALL="${OUT_TARBALL:-$REPO_ROOT/aortaseg-validate.tar.gz}"
RUN_INIMAGE_TESTS="${RUN_INIMAGE_TESTS:-1}"   # set to 0 to skip the sanity test

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
echo "[2/3] docker build -f Dockerfile.validation -t $IMAGE ."
docker build -f Dockerfile.validation -t "$IMAGE" "$REPO_ROOT"

if [ "$RUN_INIMAGE_TESTS" = "1" ]; then
    echo "      running unit tests inside the image (no GPU needed)..."
    docker run --rm --entrypoint python "$IMAGE" -m pytest -q
fi

# --- 3. export the shippable tarball ----------------------------------------
echo "[3/3] docker save $IMAGE -> $OUT_TARBALL (this takes a while)"
docker save "$IMAGE" | gzip > "$OUT_TARBALL"

echo
echo "DONE."
echo "  Bundle:  $OUT_TARBALL"
ls -lh "$OUT_TARBALL" | awk '{print "  Size:    "$5}'
echo
echo "Send $OUT_TARBALL and HANDOFF.md to your coworker."
echo "On the 4090:  docker load -i $(basename "$OUT_TARBALL")  then follow HANDOFF.md."
