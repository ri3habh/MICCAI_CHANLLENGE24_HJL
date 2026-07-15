"""
Parametrized inference runner for the Aortaseg24 nnU-Net models.

One script for both machines. Nothing is retrained or modified -- this only
chooses which frozen model(s) to run, on which device, over which paths.

Examples
--------
# Laptop smoke test (free): lightest single model on CPU
python run_inference.py --device cpu --mode single --model lowres \
    --input ./input/images/ct-angiography --output ./output/images/aortic-branches

# AWS (24 GB GPU): the real, full ensemble the authors intended
python run_inference.py --device cuda --mode ensemble \
    --input ./input/images/ct-angiography --output ./output/images/aortic-branches

Notes
-----
* --mode single runs ONE model. Valid result, marginally rougher than ensemble.
* --mode ensemble reproduces the original predict_from_folder_aorta24 exactly
  (lowres -> ResEnc-M -> ResEnc-L, then majority vote). Needs a big-VRAM GPU.
* Input folder must contain a single CT angiography .mha or .tiff volume.
* Output is written as <output>/output.mha.
"""
import argparse
import os
import sys
from pathlib import Path

import torch

# The model class + I/O + voting helpers live in new_inference_code. That module
# pulls in nnunetv2/monai/etc, so we import it lazily inside main() -- this keeps
# `python run_inference.py --help` working before the deps are installed.
nnUNetPredictor = load_image_file_as_array = write_array_as_image_file = None
vote_combine_2 = _show_torch_cuda_info = None


def _load_repo():
    global nnUNetPredictor, load_image_file_as_array, write_array_as_image_file
    global vote_combine_2, _show_torch_cuda_info
    from new_inference_code import (
        nnUNetPredictor as _p,
        load_image_file_as_array as _l,
        write_array_as_image_file as _w,
        vote_combine_2 as _v,
        _show_torch_cuda_info as _s,
    )
    nnUNetPredictor, load_image_file_as_array, write_array_as_image_file = _p, _l, _w
    vote_combine_2, _show_torch_cuda_info = _v, _s


# Model group -> (weights subfolder relative to --resources, default folds for single mode)
MODELS = {
    "lowres":  ("Dataset040_Aortaseg24/nnUNetTrainer__nnUNetPlans__3d_lowres", (0,)),
    "resencm": ("Dataset040_Aortaseg24/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres", (0,)),
    "resencl": ("Dataset040_Aortaseg24/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres", ("all",)),
}


def build_predictor(resources: Path, subfolder: str, folds, device: torch.device):
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        # class forces this False for non-cuda devices anyway
        perform_everything_on_device=(device.type == "cuda"),
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    predictor.initialize_from_trained_model_folder(
        os.path.join(str(resources), subfolder),
        use_folds=tuple(folds),
        checkpoint_name="checkpoint_final.pth",
    )
    return predictor


def run_single(img, props, resources, model, folds, device):
    subfolder, default_folds = MODELS[model]
    use_folds = folds if folds is not None else default_folds
    print(f"[single] model={model} folds={use_folds} device={device}")
    predictor = build_predictor(resources, subfolder, use_folds, device)
    ret = predictor.predict_single_npy_array(img, props, None, None, False)
    del predictor
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return ret


def run_ensemble(img, props, resources, device):
    """Faithful reproduction of the original predict_from_folder_aorta24 flow."""
    print(f"[ensemble] lowres(0) -> ResEnc-M(0,1,2) -> ResEnc-L(4,all), device={device}")

    # 1) low-res, fold 0
    p0 = build_predictor(resources, MODELS["lowres"][0], (0,), device)
    ret0 = p0.predict_single_npy_array(img, props, None, None, False)
    del p0
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 2) ResEnc-M full-res, folds 0,1,2 (prev stage = ret0, as in original)
    p2 = build_predictor(resources, MODELS["resencm"][0], (0, 1, 2), device)
    ret2 = p2.predict_single_npy_array(img, props, None, None, False, ret0)
    del p2
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 3) ResEnc-L full-res, folds 4,all (prev stage = ret2, as in original)
    p4 = build_predictor(resources, MODELS["resencl"][0], (4, "all"), device)
    ret4 = p4.predict_single_npy_array(img, props, None, None, False, ret2)
    del p4

    # Final label map = majority vote of the two full-res models (original behavior)
    ret = vote_combine_2(ret2, ret4)
    del ret2, ret4
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return ret


def parse_folds(raw):
    if raw is None:
        return None
    return [int(f) if f != "all" else "all" for f in raw]


def main():
    parser = argparse.ArgumentParser(
        description="Run Aortaseg24 nnU-Net inference (single model or full ensemble).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda",
                        help="cpu = laptop/no-VRAM (slow, safe). cuda = GPU.")
    parser.add_argument("--mode", choices=["single", "ensemble"], default="ensemble",
                        help="single = one model (laptop). ensemble = full 3-model vote (needs big GPU).")
    parser.add_argument("--model", choices=list(MODELS.keys()), default="lowres",
                        help="Which model to use in --mode single. lowres is lightest (best on 4 GB).")
    parser.add_argument("--folds", nargs="+", default=None,
                        help="Override folds for single mode, e.g. --folds 0 1 2  or  --folds all")
    parser.add_argument("--input", required=True,
                        help="Folder containing one CT angiography .mha/.tiff volume.")
    parser.add_argument("--output", required=True,
                        help="Output folder. Writes <output>/output.mha.")
    parser.add_argument("--resources", default="resources",
                        help="Folder holding the downloaded weights (Dataset040_Aortaseg24/...).")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        sys.exit("ERROR: --device cuda requested but torch.cuda.is_available() is False. "
                 "Use --device cpu, or fix your CUDA/torch install.")

    input_path = Path(args.input)
    output_path = Path(args.output)
    resources = Path(args.resources)

    if not input_path.is_dir():
        sys.exit(f"ERROR: input folder not found: {input_path}")
    if not resources.is_dir():
        sys.exit(f"ERROR: resources/weights folder not found: {resources} "
                 f"(download the weights first -- see README).")
    output_path.mkdir(parents=True, exist_ok=True)

    _load_repo()  # import the heavy repo deps now that args are validated

    device = torch.device(args.device)
    _show_torch_cuda_info()

    print(f"Loading CT from {input_path} ...")
    img, spacing, direction, origin, props, ori_axcode = load_image_file_as_array(location=input_path)

    folds = parse_folds(args.folds)
    if args.mode == "single":
        ret = run_single(img, props, resources, args.model, folds, device)
    else:
        ret = run_ensemble(img, props, resources, device)
    del img

    write_array_as_image_file(
        location=output_path,
        array=ret,
        spacing=spacing,
        direction=direction,
        origin=origin,
        ori_axcode=ori_axcode,
    )
    print(f"Saved segmentation to {output_path / 'output.mha'}")


if __name__ == "__main__":
    raise SystemExit(main())
