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


# Source names per canonical vessel, verified against the real AortaSeg24
# dataset.json (Dataset040_Aortaseg24, 24-class SVS aortic-zone scheme).
# The exact challenge class names come first; generic synonyms follow as fallbacks.
#
# Decisions baked in here:
#  - abdominal_aorta = infrarenal aorta only = "Zone9" (label 17). The other
#    abdominal zones (6/7/8, celiac->renal) are intentionally NOT included; if the
#    med team later wants the full anatomical abdominal aorta, add "Zone6"/"Zone7"/
#    "Zone8" here (they will merge into abdominal_aorta via many-to-one remap).
#  - common iliac = Zone10 (R/L), external iliac = Zone11 (R/L), per the SVS
#    aortoiliac zone extension the challenge uses.
#  - the internal-iliac class names contain a source typo ("Internal lliac"); the
#    exact string is included so it matches.
VESSEL_ALIASES: dict[str, list[str]] = {
    "abdominal_aorta": ["Zone9", "abdominal aorta", "abdominal_aorta"],
    "left_renal": ["Left Renal Artery", "left renal", "left_renal"],
    "right_renal": ["Right Renal Artery", "right renal", "right_renal"],
    "left_common_iliac": ["Zone10 L", "left common iliac", "left_common_iliac"],
    "right_common_iliac": ["Zone10 R", "right common iliac", "right_common_iliac"],
    "left_internal_iliac": ["Left Internal lliac Artery", "Left Internal Iliac Artery",
                            "left internal iliac", "left_internal_iliac"],
    "right_internal_iliac": ["Right Internal lliac Artery", "Right Internal Iliac Artery",
                             "right internal iliac", "right_internal_iliac"],
    "left_external_iliac": ["Zone11 L", "left external iliac", "left_external_iliac"],
    "right_external_iliac": ["Zone11 R", "right external iliac", "right_external_iliac"],
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
