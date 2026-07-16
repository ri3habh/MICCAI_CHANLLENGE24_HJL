"""Pair each CT image file with its ground-truth label file."""
from __future__ import annotations

from dataclasses import dataclass
from glob import glob
from pathlib import Path


@dataclass(frozen=True)
class Case:
    case_id: str
    image_path: Path
    label_path: Path


def _stem(filename: str) -> str:
    """Filename with extension removed. Treats .nii.gz as one extension."""
    if filename.endswith(".nii.gz"):
        return filename[: -len(".nii.gz")]
    return Path(filename).stem


def pair_cases(
    images_dir: Path,
    labels_dir: Path,
    image_glob: str = "*.mha",
    label_template: str = "{stem}.mha",
    image_suffix: str = "",
) -> list[Case]:
    """Match each image to its label file.

    ``{stem}`` in ``label_template`` is the image filename minus its extension.
    ``{case_id}`` is that stem with a trailing ``image_suffix`` removed, letting
    image and label use different suffixes on a shared id -- e.g. images named
    ``subject001_CTA.nii.gz`` and labels ``subject001_label.seg.nrrd`` pair with
    ``image_glob="*_CTA.nii.gz"``, ``image_suffix="_CTA"``,
    ``label_template="{case_id}_label.seg.nrrd"``. With the default empty
    ``image_suffix``, ``{case_id}`` equals ``{stem}``.
    """
    images_dir, labels_dir = Path(images_dir), Path(labels_dir)
    image_paths = sorted(Path(p) for p in glob(str(images_dir / image_glob)))

    cases: list[Case] = []
    missing: list[str] = []
    for img in image_paths:
        stem = _stem(img.name)
        case_id = stem[: -len(image_suffix)] if image_suffix and stem.endswith(image_suffix) else stem
        label_path = labels_dir / label_template.format(stem=stem, case_id=case_id)
        if not label_path.is_file():
            missing.append(f"{img.name} -> expected {label_path}")
            continue
        cases.append(Case(case_id=case_id, image_path=img, label_path=label_path))

    if missing:
        raise FileNotFoundError(
            "No label file found for the following image(s):\n  " + "\n  ".join(missing)
        )
    return sorted(cases, key=lambda c: c.case_id)
