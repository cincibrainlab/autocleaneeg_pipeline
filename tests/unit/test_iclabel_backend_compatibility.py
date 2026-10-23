"""Exercise the real ICLabel models across supported inference backends."""

import numpy as np
import pytest
from mne_icalabel.iclabel import network


def test_iclabel_torch_and_onnx_fallback_agree(monkeypatch):
    """Security upgrades must preserve inference and the Intel-Mac fallback."""
    pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")
    rng = np.random.default_rng(42)
    features = (
        rng.uniform(-1, 1, (32, 32, 1, 3)).astype(np.float32),
        rng.uniform(-1, 1, (1, 100, 1, 3)).astype(np.float32),
        rng.uniform(-1, 1, (1, 100, 1, 3)).astype(np.float32),
    )
    torch_scores = network.run_iclabel(*features, backend="torch")
    onnx_scores = network.run_iclabel(*features, backend="onnx")
    assert torch_scores.shape == (3, 7)
    assert np.isfinite(torch_scores).all()
    np.testing.assert_allclose(torch_scores.sum(axis=1), 1, atol=1e-5)
    np.testing.assert_allclose(torch_scores, onnx_scores, rtol=1e-4, atol=1e-6)

    original_import = network.import_optional_dependency

    def without_torch(name, **kwargs):
        return None if name == "torch" else original_import(name, **kwargs)

    monkeypatch.setattr(network, "import_optional_dependency", without_torch)
    fallback_scores = network.run_iclabel(*features, backend=None)
    np.testing.assert_allclose(fallback_scores, onnx_scores, rtol=1e-5, atol=1e-6)
