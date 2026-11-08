import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[2] / "experiments" / "m2_calibrated_cost.py"
SPEC = importlib.util.spec_from_file_location("m2_calibrated_cost", SCRIPT)
M2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M2)


def test_failure_mix_uses_only_labels_supplied_by_training_fold():
    train_labels = np.array([0] * 80 + [1] * 2 + [2] * 3 + [3] * 5 + [4] * 10)
    held_out_labels = np.array([0] * 20 + [1] * 10 + [2] * 1 + [3] * 1 + [4] * 1)

    mix = M2.failure_mix_from_labels(train_labels)

    expected = np.array([2, 3, 5, 10], dtype=float) / 20
    assert np.allclose(mix, expected)
    assert not np.allclose(mix, M2.failure_mix_from_labels(np.concatenate([train_labels, held_out_labels])))


def test_valmix_calibration_uses_explicit_fold_mix():
    raw = np.array([[0.8, 0.1, 0.05, 0.03, 0.02]])
    fold_mix = np.array([0.1, 0.2, 0.3, 0.4])

    calibrated = M2.calibrated_probs(raw, 1.0, -2.0, "valmix", severity_mix=fold_mix)

    assert calibrated.shape == (1, 5)
    assert np.allclose(calibrated[0, 1:] / calibrated[0, 1:].sum(), fold_mix)
    assert np.isclose(calibrated.sum(), 1.0)
