"""Cached, resumable single-case inference using the frozen ensemble (unmodified)."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_case_array(image_path: Path):
    """Explicit-path twin of new_inference_code.load_image_file_as_array.

    Same preprocessing (SimpleITKIO read + the two axis flips) so the model sees
    exactly what it expects, but accepts any SimpleITK-readable single file.
    """
    import SimpleITK
    from monai import transforms
    from monai.data import ITKReader
    from nibabel.orientations import aff2axcodes
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

    image_path = str(image_path)
    result = SimpleITK.ReadImage(image_path)
    spacing = result.GetSpacing()
    direction = result.GetDirection()
    origin = result.GetOrigin()
    _, meta_data = transforms.LoadImage(reader=ITKReader())(image_path)
    img, props = SimpleITKIO().read_images([image_path])
    ori_axcode = aff2axcodes(meta_data["affine"])
    img = np.flip(img, axis=2)
    img = np.flip(img, axis=3)
    return img, spacing, direction, origin, props, ori_axcode


def predict_case(case, resources, predictions_dir, device,
                 ensemble_fn=None, loader=None, writer=None):
    predictions_dir = Path(predictions_dir)
    out_dir = predictions_dir / case.case_id
    out_file = out_dir / "output.mha"
    if out_file.is_file():
        return out_file  # resumable: already predicted

    if ensemble_fn is None:
        from run_inference import run_ensemble as ensemble_fn  # noqa
    if loader is None:
        loader = load_case_array
    if writer is None:
        from new_inference_code import write_array_as_image_file as writer  # noqa

    img, spacing, direction, origin, props, ori_axcode = loader(case.image_path)
    ret = ensemble_fn(img, props, Path(resources), device)
    del img
    writer(location=out_dir, array=ret, spacing=spacing, origin=origin,
           direction=direction, ori_axcode=ori_axcode)
    return out_file
