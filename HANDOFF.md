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
You have the 100 CTAs and their label masks. Put (or mount) them as two folders:
- images: one CT per file (`.mha` or `.nii.gz`)
- labels: the matching ground-truth mask per case

If your filenames are e.g. `subject001_CTA.mha` / `subject001_CTA_label.mha`, add
`--image-glob "*_CTA.mha" --label-template "{stem}_label.mha"` to the commands below.

## 2. Preflight (takes seconds — do this first)
```
docker run --rm --gpus all \
  -v /ABS/PATH/images:/data/images \
  -v /ABS/PATH/labels:/data/labels \
  -v /ABS/PATH/out:/out \
  aortaseg-validate --preflight --images /data/images --labels /data/labels --outdir /out
```
It must print `preflight PASSED`. If not, fix what it reports (GPU not visible,
wrong folder, filename mismatch) before continuing.

## 3. Run the full batch (hours — it prints progress + ETA)
Same command, drop `--preflight`:
```
docker run --rm --gpus all \
  -v /ABS/PATH/images:/data/images \
  -v /ABS/PATH/labels:/data/labels \
  -v /ABS/PATH/out:/out \
  aortaseg-validate --images /data/images --labels /data/labels --outdir /out
```
Safe to stop and rerun: finished cases are cached and skipped. If the machine
reboots, just run the same command again.

## 4. Send back
Zip and return only the small `out/results/` folder:
`per_case_vessel.csv`, `summary.csv`, `summary.md`, `failures.log`.
(The big `out/predictions/` folder can stay on your machine.)
```
