from validation.run_validation import build_parser, preflight


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
