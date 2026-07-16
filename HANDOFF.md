# Running the AortaSeg24 Validation (coworker guide)

You need: a machine with an NVIDIA GPU (e.g. RTX 4090), Docker, and the NVIDIA
Container Toolkit (so `--gpus all` works). For the one-time build, the machine
also needs to reach Docker Hub to pull the CUDA PyTorch base image (~5 GB).
You do **not** need to set up Python or nnU-Net yourself — the build handles it.

You received: this **repo** and the **weights zip** (`resources...`, ~3.2 GB).

## 0. Build the image (one time, ~10-20 min)
From the repo root, point the script at the weights zip and run it:
```
WEIGHTS_ZIP=/ABS/PATH/to/weights.zip ./build_bundle.sh
```
This unpacks the weights, builds the `aortaseg-validate` image (weights baked
in), and runs its unit tests to confirm the image is sane. When it prints
`DONE. Image 'aortaseg-validate' is built`, you're ready. You only do this once.

## 1. Point at your data
This is the `ResampledDataInNifti` dataset: 100 CTAs and their masks in two folders:
- `ResampledDataInNifti/images/` — one CT per file, `subjectNNN_CTA.nii.gz`
- `ResampledDataInNifti/masks/`  — one mask per case, `subjectNNN_label.seg.nrrd`

The commands below already carry the flags for this layout:
`--image-glob "*_CTA.nii.gz" --image-suffix "_CTA" --label-template "{case_id}_label.seg.nrrd"`
(`{case_id}` is the image name with `_CTA` stripped, so `subject001_CTA` pairs with
`subject001_label.seg.nrrd`). If the folder names differ, only the `-v` mount paths change.

## 2. Preflight (takes seconds — do this first)
```
docker run --rm --gpus all \
  -v /ABS/PATH/ResampledDataInNifti/images:/data/images \
  -v /ABS/PATH/ResampledDataInNifti/masks:/data/labels \
  -v /ABS/PATH/out:/out \
  aortaseg-validate --preflight \
    --images /data/images --labels /data/labels --outdir /out \
    --image-glob "*_CTA.nii.gz" --image-suffix "_CTA" --label-template "{case_id}_label.seg.nrrd"
```
It must print `preflight PASSED`. If not, fix what it reports (GPU not visible,
wrong folder, filename mismatch) before continuing.

## 2b. Smoke-test 2 cases first (a few minutes)
Before the multi-hour run, prove it end-to-end on 2 cases by adding `--limit 2`
(drop `--preflight`, keep everything else):
```
docker run --rm --gpus all \
  -v /ABS/PATH/ResampledDataInNifti/images:/data/images \
  -v /ABS/PATH/ResampledDataInNifti/masks:/data/labels \
  -v /ABS/PATH/out:/out \
  aortaseg-validate --limit 2 \
    --images /data/images --labels /data/labels --outdir /out \
    --image-glob "*_CTA.nii.gz" --image-suffix "_CTA" --label-template "{case_id}_label.seg.nrrd"
```
Open `out/results/summary.md` — if 2 cases scored with sane numbers, continue.

## 3. Run the full batch (hours — it prints progress + ETA)
Same command, drop `--limit 2`:
```
docker run --rm --gpus all \
  -v /ABS/PATH/ResampledDataInNifti/images:/data/images \
  -v /ABS/PATH/ResampledDataInNifti/masks:/data/labels \
  -v /ABS/PATH/out:/out \
  aortaseg-validate \
    --images /data/images --labels /data/labels --outdir /out \
    --image-glob "*_CTA.nii.gz" --image-suffix "_CTA" --label-template "{case_id}_label.seg.nrrd"
```
Safe to stop and rerun: finished cases are cached and skipped. If the machine
reboots, just run the same command again.

## 4. Send back
Zip and return only the small `out/results/` folder:
`per_case_vessel.csv`, `summary.csv`, `summary.md`, `failures.log`.
(The big `out/predictions/` folder can stay on your machine.)
```
