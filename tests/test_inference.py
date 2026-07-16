from pathlib import Path
import numpy as np
from validation.inference import predict_case
from validation.pairing import Case


def _fake_loader(image_path):
    img = np.zeros((1, 4, 4, 4), dtype=np.float32)
    return img, (1.0, 1.0, 1.0), (1, 0, 0, 0, 1, 0, 0, 0, 1), (0, 0, 0), {}, ("R", "A", "S")


def test_predict_runs_then_caches(tmp_path):
    calls = {"n": 0}

    def fake_ensemble(img, props, resources, device):
        calls["n"] += 1
        return np.ones((4, 4, 4), dtype=np.uint8)

    written = {}

    def fake_writer(*, location, array, spacing, origin, direction, ori_axcode):
        location.mkdir(parents=True, exist_ok=True)
        (location / "output.mha").write_bytes(b"x")
        written["array"] = array

    case = Case("c1", tmp_path / "c1.mha", tmp_path / "c1_lab.mha")
    preds = tmp_path / "predictions"

    out1 = predict_case(case, tmp_path, preds, device="cpu",
                        ensemble_fn=fake_ensemble, loader=_fake_loader, writer=fake_writer)
    assert out1 == preds / "c1" / "output.mha"
    assert out1.is_file()
    assert calls["n"] == 1

    # second call: file exists -> no re-inference
    out2 = predict_case(case, tmp_path, preds, device="cpu",
                        ensemble_fn=fake_ensemble, loader=_fake_loader, writer=fake_writer)
    assert out2 == out1
    assert calls["n"] == 1  # unchanged
