# Running the AortaSeg24 Validation (coworker guide)

You need: a machine with an NVIDIA GPU (e.g. RTX 4090), Docker, and the NVIDIA
Container Toolkit (so `--gpus all` works). You do **not** need Python, nnU-Net,
or any model download — the weights are already inside the image.

## 0. Load the image
You received `aortaseg-validate.tar.gz`. Load it:
```
docker load -i aortaseg-validate.tar.gz
```

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

## 3. Run the full batch (hours — it prints progress + ETA)
Same command, drop `--preflight`:
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
