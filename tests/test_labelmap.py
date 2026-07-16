import numpy as np
from validation.labelmap import (
    VESSELS, remap_volume, build_model_mapping, canonical_index, VESSEL_ALIASES,
)


def test_vessels_order_and_count():
    assert len(VESSELS) == 9
    assert VESSELS[0] == "abdominal_aorta"
    assert canonical_index("right_external_iliac") == 9


def test_remap_collapses_unknown_to_background():
    vol = np.array([[0, 1, 2], [3, 99, 5]], dtype=np.int16)
    mapping = {1: 1, 2: 2, 5: 4}  # 3 and 99 unmapped -> 0
    out = remap_volume(vol, mapping)
    assert out.tolist() == [[0, 1, 2], [0, 0, 4]]


def test_remap_many_to_one():
    vol = np.array([10, 11, 12, 0], dtype=np.int16)
    mapping = {10: 1, 11: 1, 12: 1}  # three model zones -> abdominal_aorta
    assert remap_volume(vol, mapping).tolist() == [1, 1, 1, 0]


def test_build_model_mapping_matches_names_case_insensitively():
    dataset_json = {
        "labels": {
            "background": 0,
            "Right Renal Artery": 7,
            "left_renal": 8,
            "Some Aortic Zone We Ignore": 3,
        }
    }
    aliases = {
        "right_renal": ["right renal artery", "right_renal"],
        "left_renal": ["left renal artery", "left_renal"],
    }
    mapping = build_model_mapping(dataset_json, aliases)
    assert mapping[7] == canonical_index("right_renal")   # 3
    assert mapping[8] == canonical_index("left_renal")    # 2
    assert 3 not in mapping  # unmatched model class excluded
