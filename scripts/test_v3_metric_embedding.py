"""Dependency-free V3.3 metric embedding smoke test."""

from pathlib import Path
import sys
import tempfile

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adaptive_gesture.learning.metric_embedding import MetricEmbeddingBank


def main() -> int:
    rng = np.random.default_rng(123)
    groups = {}
    for label, center in enumerate((-1.0, 0.2, 1.2)):
        groups[f"gesture_{label}"] = [
            (np.full(16, center, dtype=np.float32)
             + rng.normal(0.0, 0.08, 16).astype(np.float32))
            for _ in range(8)
        ]

    bank = MetricEmbeddingBank(embedding_dim=8)
    reports = bank.fit_from_labeled_groups(groups, hidden_dim=24, seed=17)
    if len(reports) != 1:
        raise RuntimeError("Expected one trained metric encoder.")

    vector = groups["gesture_0"][0]
    embedded = bank.encode(vector)
    if embedded.shape != (8,):
        raise RuntimeError(f"Unexpected embedding shape: {embedded.shape}")
    if not np.isclose(np.linalg.norm(embedded), 1.0, atol=1e-4):
        raise RuntimeError("Embedding is not L2-normalized.")

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "metric.npz"
        bank.save(path)
        restored = MetricEmbeddingBank.load(path)
        np.testing.assert_allclose(restored.encode(vector), embedded, atol=1e-6)

    report = reports[0]
    print("V3.3 learned metric embedding smoke test: PASS")
    print(
        f"input={report.input_dimension}D output={report.embedding_dimension}D "
        f"classes={report.class_count} separation={report.separation_ratio:.2f}x"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
