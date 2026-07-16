import numpy as np
from validation.labelmap import (
    VESSELS, remap_volume, build_model_mapping, canonical_index, VESSEL_ALIASES,
    unmapped_vessels,
)


def test_vessels_order_and_count():
    assert len(VESSELS) == 9
    assert VESSELS[0] == "abdominal_aorta"
    assert canonical_index("right_external_iliac") == 9


# The real AortaSeg24 label scheme (Dataset040_Aortaseg24/dataset.json), verified
# against the downloaded weights. Pins the default VESSEL_ALIASES so a change that
# breaks resolution of any of the 9 Toralis vessels fails loudly.
AORTASEG24_LABELS = {
    "background": 0, "Zone0": 1, "Innominate": 2, "Zone1": 3,
    "Left Common Carotid": 4, "Zone2": 5, "Left Subclavian Artery": 6, "Zone3": 7,
    "Zone4": 8, "Zone5": 9, "Zone6": 10, "Celiac Artery": 11, "Zone7": 12,
    "SMA": 13, "Zone8": 14, "Right Renal Artery": 15, "Left Renal Artery": 16,
    "Zone9": 17, "Zone10 R": 18, "Zone10 L": 19, "Right Internal lliac Artery": 20,
    "Left Internal lliac Artery": 21, "Zone11 R": 22, "Zone11 L": 23,
}


def test_default_aliases_resolve_all_nine_on_real_labels():
    mapping = build_model_mapping({"labels": AORTASEG24_LABELS})
    assert unmapped_vessels(mapping) == []
    # abdominal_aorta = infrarenal only (Zone9=17); thoracic/branch zones dropped.
    assert mapping == {
        17: canonical_index("abdominal_aorta"),
        16: canonical_index("left_renal"),
        15: canonical_index("right_renal"),
        19: canonical_index("left_common_iliac"),
        18: canonical_index("right_common_iliac"),
        21: canonical_index("left_internal_iliac"),
        20: canonical_index("right_internal_iliac"),
        23: canonical_index("left_external_iliac"),
        22: canonical_index("right_external_iliac"),
    }
    # Zone6/7/8, celiac, SMA, thoracic zones must NOT leak into abdominal_aorta.
    assert 10 not in mapping and 11 not in mapping and 13 not in mapping


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


def test_unmapped_vessels_reports_missing_names():
    mapping = {10: 1, 11: 2}  # only abdominal_aorta and left_renal covered
    missing = unmapped_vessels(mapping)
    assert set(missing) == set(VESSELS) - {"abdominal_aorta", "left_renal"}
    assert len(missing) == 7


def test_unmapped_vessels_empty_when_all_covered():
    mapping = {i: i for i in range(1, 10)}  # model idx == canonical idx, full coverage
    assert unmapped_vessels(mapping) == []
