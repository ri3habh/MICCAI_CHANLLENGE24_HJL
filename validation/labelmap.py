"""Map model output classes and ground-truth labels into a shared 9-vessel space.

Canonical label space: 0 = background, 1..9 = VESSELS in order.
"""
from __future__ import annotations

import numpy as np

VESSELS: list[str] = [
    "abdominal_aorta",       # -> 1
    "left_renal",            # -> 2
    "right_renal",           # -> 3
    "left_common_iliac",     # -> 4
    "right_common_iliac",    # -> 5
    "left_internal_iliac",   # -> 6
    "right_internal_iliac",  # -> 7
    "left_external_iliac",   # -> 8
    "right_external_iliac",  # -> 9
]


def canonical_index(name: str) -> int:
    return VESSELS.index(name) + 1


def _norm(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


# Known source names / synonyms per canonical vessel. Extend after inspecting the
# real dataset.json in Step 6. abdominal_aorta may match several aortic-zone classes.
VESSEL_ALIASES: dict[str, list[str]] = {
    "abdominal_aorta": ["abdominal aorta", "abdominal_aorta", "aorta abdominal"],
    "left_renal": ["left renal", "left renal artery", "left_renal"],
    "right_renal": ["right renal", "right renal artery", "right_renal"],
    "left_common_iliac": ["left common iliac", "left_common_iliac"],
    "right_common_iliac": ["right common iliac", "right_common_iliac"],
    "left_internal_iliac": ["left internal iliac", "left_internal_iliac"],
    "right_internal_iliac": ["right internal iliac", "right_internal_iliac"],
    "left_external_iliac": ["left external iliac", "left_external_iliac"],
    "right_external_iliac": ["right external iliac", "right_external_iliac"],
}

# Ground-truth integer -> canonical index. VERIFY against a real label file (Step 6);
# these values are the AortaSeg24 default guess and MUST be confirmed before the batch.
GT_CLASS_TO_CANONICAL: dict[int, int] = {}


def remap_volume(vol: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    out = np.zeros_like(vol)
    for src, dst in mapping.items():
        out[vol == src] = dst
    return out


def build_model_mapping(dataset_json: dict, aliases: dict[str, list[str]] = VESSEL_ALIASES) -> dict[int, int]:
    alias_lookup: dict[str, int] = {}
    for vessel, names in aliases.items():
        idx = canonical_index(vessel)
        for n in names:
            alias_lookup[_norm(n)] = idx

    mapping: dict[int, int] = {}
    for name, model_idx in dataset_json["labels"].items():
        if int(model_idx) == 0:
            continue
        canon = alias_lookup.get(_norm(name))
        if canon is not None:
            mapping[int(model_idx)] = canon
    return mapping


def unmapped_vessels(mapping: dict[int, int]) -> list[str]:
    """Canonical vessel names whose 1-based canonical index is missing from mapping.values()."""
    covered = set(mapping.values())
    return [v for v in VESSELS if canonical_index(v) not in covered]
