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
) -> list[Case]:
    images_dir, labels_dir = Path(images_dir), Path(labels_dir)
    image_paths = sorted(Path(p) for p in glob(str(images_dir / image_glob)))

    cases: list[Case] = []
    missing: list[str] = []
    for img in image_paths:
        stem = _stem(img.name)
        label_path = labels_dir / label_template.format(stem=stem)
        if not label_path.is_file():
            missing.append(f"{img.name} -> expected {label_path}")
            continue
        cases.append(Case(case_id=stem, image_path=img, label_path=label_path))

    if missing:
        raise FileNotFoundError(
            "No label file found for the following image(s):\n  " + "\n  ".join(missing)
        )
    return sorted(cases, key=lambda c: c.case_id)
