from pathlib import Path
import pytest
from validation.pairing import pair_cases, Case


def _touch(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")


def test_pairs_matching_stems(tmp_path):
    imgs, labs = tmp_path / "img", tmp_path / "lab"
    _touch(imgs / "subject002.mha")
    _touch(imgs / "subject001.mha")
    _touch(labs / "subject001.mha")
    _touch(labs / "subject002.mha")
    cases = pair_cases(imgs, labs)
    assert [c.case_id for c in cases] == ["subject001", "subject002"]
    assert cases[0].image_path == imgs / "subject001.mha"
    assert cases[0].label_path == labs / "subject001.mha"


def test_strips_double_extension(tmp_path):
    imgs, labs = tmp_path / "img", tmp_path / "lab"
    _touch(imgs / "caseA.nii.gz")
    _touch(labs / "caseA.nii.gz")
    cases = pair_cases(imgs, labs, image_glob="*.nii.gz", label_template="{stem}.nii.gz")
    assert cases[0].case_id == "caseA"


def test_custom_label_template(tmp_path):
    imgs, labs = tmp_path / "img", tmp_path / "lab"
    _touch(imgs / "subject001_CTA.mha")
    _touch(labs / "subject001_CTA_label.mha")
    cases = pair_cases(imgs, labs, label_template="{stem}_label.mha")
    assert cases[0].label_path == labs / "subject001_CTA_label.mha"


def test_image_suffix_maps_cta_to_label(tmp_path):
    # Real ResampledDataInNifti convention: subjectNNN_CTA.nii.gz <-> subjectNNN_label.seg.nrrd
    imgs, labs = tmp_path / "images", tmp_path / "masks"
    _touch(imgs / "subject001_CTA.nii.gz")
    _touch(labs / "subject001_label.seg.nrrd")
    cases = pair_cases(imgs, labs, image_glob="*_CTA.nii.gz",
                       image_suffix="_CTA", label_template="{case_id}_label.seg.nrrd")
    assert len(cases) == 1
    assert cases[0].case_id == "subject001"
    assert cases[0].image_path == imgs / "subject001_CTA.nii.gz"
    assert cases[0].label_path == labs / "subject001_label.seg.nrrd"


def test_missing_label_raises(tmp_path):
    imgs, labs = tmp_path / "img", tmp_path / "lab"
    _touch(imgs / "subject001.mha")
    with pytest.raises(FileNotFoundError) as e:
        pair_cases(imgs, labs)
    assert "subject001" in str(e.value)
