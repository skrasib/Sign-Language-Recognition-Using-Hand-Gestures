"""Train/retrain the frozen V3.3 metric encoder from raw hybrid gesture memory.

Normal app startup trains once automatically if no encoder exists.  This script
exists for controlled research runs where the source memory is intentionally
chosen and the encoder should be regenerated explicitly.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from adaptive_gesture.learning.metric_embedding import MetricEmbeddingBank
from adaptive_gesture.learning.metric_runtime import load_source_learner


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=PROJECT_ROOT / "data" / "v3" / "gesture_memory_hybrid.json",
        help="Raw hybrid gesture-memory JSON used as metric-training source.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "v3" / "metric_encoder_v33.npz",
        help="Frozen metric encoder output.",
    )
    parser.add_argument("--embedding-dim", type=int, default=48)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    learner, count = load_source_learner(args.source)
    print(f"Loaded {count} source gestures from: {args.source}")
    bank, reports = MetricEmbeddingBank.train_from_learner(
        learner,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
    )

    if not reports:
        print(
            "No encoder was trained. Each feature dimension needs at least "
            "two gesture classes with at least two samples per class."
        )
        return 2

    bank.save(args.output)
    print(f"Saved frozen metric encoder: {args.output}")
    for report in reports:
        print(
            f"  {report.input_dimension}D -> {report.embedding_dimension}D | "
            f"classes={report.class_count} samples={report.sample_count} "
            f"epochs={report.epochs_ran} loss={report.final_loss:.5f} "
            f"LOO(train-only)={report.leave_one_out_accuracy:.1%} "
            f"separation={report.separation_ratio:.2f}x"
        )
    print(
        "Note: training-only geometry is diagnostic, not publication accuracy. "
        "Use disjoint source/target data for formal evaluation."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
