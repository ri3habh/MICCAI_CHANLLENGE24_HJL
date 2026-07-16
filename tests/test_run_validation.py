import json

from validation.run_validation import build_parser, preflight, run, _resencl_dataset_json


def test_parser_requires_images_labels_outdir():
    p = build_parser()
    args = p.parse_args(["--images", "a", "--labels", "b", "--outdir", "c"])
    assert args.device == "cuda"
    assert args.weak_threshold == 0.80


def test_preflight_fails_on_missing_weights(tmp_path, capsys):
    p = build_parser()
    args = p.parse_args([
        "--images", str(tmp_path), "--labels", str(tmp_path),
        "--outdir", str(tmp_path), "--resources", str(tmp_path / "nope"),
        "--device", "cpu",
    ])
    rc = preflight(args)
    assert rc != 0
    assert "weights" in capsys.readouterr().out.lower()


def _write_dataset_json(resources, labels: dict) -> None:
    ds = _resencl_dataset_json(resources)
    ds.parent.mkdir(parents=True, exist_ok=True)
    ds.write_text(json.dumps({"labels": labels}), encoding="utf-8")


def test_preflight_fails_on_incomplete_model_mapping(tmp_path, capsys):
    resources = tmp_path / "resources"
    # Only "abdominal aorta" maps to a canonical vessel; the other 8 are missing.
    _write_dataset_json(resources, {"background": 0, "abdominal aorta": 1})

    p = build_parser()
    args = p.parse_args([
        "--images", str(tmp_path), "--labels", str(tmp_path),
        "--outdir", str(tmp_path), "--resources", str(resources),
        "--device", "cpu",
    ])
    rc = preflight(args)
    assert rc == 6
    out = capsys.readouterr().out.lower()
    assert "incomplete" in out
    assert "left_renal" in out or "left renal" in out


def test_run_returns_6_before_loading_repo_when_mapping_incomplete(tmp_path, monkeypatch):
    resources = tmp_path / "resources"
    _write_dataset_json(resources, {"background": 0, "abdominal aorta": 1})

    # GT_CLASS_TO_CANONICAL must be non-empty so we reach the mapping gate (not the gt gate).
    monkeypatch.setattr("validation.labelmap.GT_CLASS_TO_CANONICAL", {1: 1})

    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()

    def _boom():
        raise AssertionError("_load_repo() should not be called when mapping is incomplete")

    monkeypatch.setattr("run_inference._load_repo", _boom)

    p = build_parser()
    args = p.parse_args([
        "--images", str(images), "--labels", str(labels),
        "--outdir", str(tmp_path / "out"), "--resources", str(resources),
        "--device", "cpu",
    ])
    rc = run(args)
    assert rc == 6
