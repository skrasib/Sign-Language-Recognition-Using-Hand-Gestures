"""Runtime helpers for V3.3 frozen metric embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from adaptive_gesture.learning.metric_embedding import (
    MetricEmbeddingBank,
    MetricTrainingReport,
)
from adaptive_gesture.learning.online_learner import OnlineGestureLearner
from adaptive_gesture.storage.gesture_store import GestureStore


@dataclass(frozen=True)
class MetricBootstrapResult:
    bank: MetricEmbeddingBank
    source_learner: OnlineGestureLearner
    source_gesture_count: int
    trained_now: bool
    reports: tuple[MetricTrainingReport, ...]


def load_source_learner(path: str | Path) -> tuple[OnlineGestureLearner, int]:
    learner = OnlineGestureLearner()
    count = GestureStore(path).load_into(learner)
    return learner, count


def load_or_train_metric_bank(
    *,
    source_memory_path: str | Path,
    encoder_path: str | Path,
    embedding_dim: int = 48,
    hidden_dim: int = 96,
    seed: int = 42,
) -> MetricBootstrapResult:
    """Load a frozen bank, or bootstrap it once from existing raw V3 memory."""
    source_learner, source_count = load_source_learner(source_memory_path)
    encoder_path = Path(encoder_path)

    if encoder_path.exists():
        bank = MetricEmbeddingBank.load(encoder_path)
        return MetricBootstrapResult(
            bank=bank,
            source_learner=source_learner,
            source_gesture_count=source_count,
            trained_now=False,
            reports=(),
        )

    bank, reports = MetricEmbeddingBank.train_from_learner(
        source_learner,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        seed=seed,
    )
    if bank.is_active:
        bank.save(encoder_path)

    return MetricBootstrapResult(
        bank=bank,
        source_learner=source_learner,
        source_gesture_count=source_count,
        trained_now=bank.is_active,
        reports=tuple(reports),
    )


def migrate_source_memory_to_metric(
    source_learner,
    target_learner,
    bank: MetricEmbeddingBank,
) -> int:
    """Encode raw hybrid memory into a new metric-space runtime learner."""
    restored = 0
    for name, source in source_learner.gestures.items():
        encoded_samples = [bank.encode(sample) for sample in source.samples]
        if not encoded_samples:
            continue
        target_learner.learn_gesture(
            name=name,
            samples=encoded_samples,
            hand_signature=source.hand_signature,
        )
        for negative in source.hard_negatives:
            target_learner.add_hard_negative(name, bank.encode(negative))
        restored += 1
    return restored
