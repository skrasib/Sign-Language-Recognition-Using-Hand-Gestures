import numpy as np

from adaptive_gesture.learning.metric_embedding import MetricEmbeddingBank, NumpyMetricMLP
from adaptive_gesture.learning.metric_runtime import migrate_source_memory_to_metric
from adaptive_gesture.learning.evt_open_set import EVTOpenSetGestureLearner
from adaptive_gesture.learning.online_learner import OnlineGestureLearner


def _clusters(seed=7, dim=12):
    rng = np.random.default_rng(seed)
    centers = [
        np.full(dim, -1.0, dtype=np.float32),
        np.full(dim, 0.2, dtype=np.float32),
        np.full(dim, 1.2, dtype=np.float32),
    ]
    xs, ys = [], []
    for label, center in enumerate(centers):
        for _ in range(10):
            xs.append(center + rng.normal(0, 0.08, dim).astype(np.float32))
            ys.append(label)
    return np.stack(xs), np.asarray(ys, dtype=np.int64)


def test_metric_encoder_learns_compact_embedding():
    x, y = _clusters()
    encoder = NumpyMetricMLP(input_dim=x.shape[1], hidden_dim=32, embedding_dim=8, seed=9)
    report = encoder.fit(x, y, epochs=180, pair_count=160, patience=30)
    embedded = encoder.encode_batch(x)

    assert embedded.shape == (30, 8)
    np.testing.assert_allclose(np.linalg.norm(embedded, axis=1), 1.0, atol=1e-4)
    assert report.leave_one_out_accuracy >= 0.90
    assert report.mean_different_distance > report.mean_same_distance


def test_metric_bank_round_trip(tmp_path):
    x, y = _clusters(dim=9)
    groups = {
        f"g{label}": [x[i] for i in range(len(y)) if y[i] == label]
        for label in np.unique(y)
    }
    bank = MetricEmbeddingBank(embedding_dim=7)
    reports = bank.fit_from_labeled_groups(groups, hidden_dim=24, seed=11)
    assert len(reports) == 1

    path = tmp_path / "metric.npz"
    bank.save(path)
    restored = MetricEmbeddingBank.load(path)

    np.testing.assert_allclose(restored.encode(x[0]), bank.encode(x[0]), atol=1e-6)
    assert restored.active_dimensions == (9,)


def test_bank_falls_back_for_untrained_dimension():
    bank = MetricEmbeddingBank(embedding_dim=6)
    vector = np.arange(5, dtype=np.float32)
    np.testing.assert_allclose(bank.encode(vector), vector)


def test_migration_encodes_runtime_memory():
    x, y = _clusters(dim=10)
    source = OnlineGestureLearner()
    for label in np.unique(y):
        source.learn_gesture(
            f"g{label}",
            [x[i] for i in range(len(y)) if y[i] == label],
            hand_signature="Right",
        )

    bank, reports = MetricEmbeddingBank.train_from_learner(
        source,
        embedding_dim=6,
        hidden_dim=24,
        seed=5,
    )
    assert reports

    target = EVTOpenSetGestureLearner(evt_min_negatives=2)
    count = migrate_source_memory_to_metric(source, target, bank)
    assert count == 3
    assert all(gesture.feature_dimension == 6 for gesture in target.gestures.values())
