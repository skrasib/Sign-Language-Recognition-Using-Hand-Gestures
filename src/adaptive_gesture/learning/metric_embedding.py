"""Lightweight learned metric embeddings for the V3.3 research branch.

This module intentionally keeps the runtime learner prototype/exemplar based.
Only the feature representation is learned.  The encoder is trained separately,
persisted, then frozen during normal runtime class addition and prediction.

The implementation is NumPy-only so the desktop application does not acquire a
large deep-learning runtime dependency.  The small MLP is trained with a
supervised contrastive pair objective: examples of the same gesture are pulled
together, while examples of different gestures are pushed apart in a
unit-normalized embedding space.

For publication-quality experiments the encoder should be trained on a source
corpus/users/classes that are disjoint from the target few-shot evaluation set.
Training from the local V3.2 gesture memory is provided as a convenient research
bootstrap, not as evidence of unseen-class generalization.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


_EPS = 1e-8


@dataclass(frozen=True)
class MetricTrainingReport:
    input_dimension: int
    class_count: int
    sample_count: int
    hidden_dimension: int
    embedding_dimension: int
    epochs_ran: int
    final_loss: float
    leave_one_out_accuracy: float
    mean_same_distance: float
    mean_different_distance: float

    @property
    def separation_ratio(self) -> float:
        return self.mean_different_distance / max(self.mean_same_distance, _EPS)


class NumpyMetricMLP:
    """Two-layer metric encoder with tanh non-linearity and L2 output norm."""

    FORMAT_VERSION = 1

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 96,
        embedding_dim: int = 48,
        seed: int = 42,
    ) -> None:
        if input_dim <= 0 or hidden_dim <= 0 or embedding_dim <= 1:
            raise ValueError("Encoder dimensions must be positive.")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.embedding_dim = int(embedding_dim)
        self.seed = int(seed)

        rng = np.random.default_rng(self.seed)
        self.input_mean = np.zeros(self.input_dim, dtype=np.float32)
        self.input_std = np.ones(self.input_dim, dtype=np.float32)

        # Xavier-like initialisation is sufficient for this compact tanh MLP.
        w1_scale = np.sqrt(2.0 / (self.input_dim + self.hidden_dim))
        w2_scale = np.sqrt(2.0 / (self.hidden_dim + self.embedding_dim))
        self.w1 = rng.normal(
            0.0, w1_scale, size=(self.input_dim, self.hidden_dim)
        ).astype(np.float32)
        self.b1 = np.zeros(self.hidden_dim, dtype=np.float32)
        self.w2 = rng.normal(
            0.0, w2_scale, size=(self.hidden_dim, self.embedding_dim)
        ).astype(np.float32)
        self.b2 = np.zeros(self.embedding_dim, dtype=np.float32)
        self.trained = False

    # ------------------------------------------------------------------
    # Forward / backward helpers
    # ------------------------------------------------------------------

    def _standardize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.input_mean) / self.input_std

    @staticmethod
    def _normalize_rows(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        norms = np.sqrt(np.sum(raw * raw, axis=1, keepdims=True) + _EPS)
        return raw / norms, norms

    def _forward_standardized(
        self,
        x_std: np.ndarray,
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        hidden_pre = x_std @ self.w1 + self.b1
        hidden = np.tanh(hidden_pre)
        raw = hidden @ self.w2 + self.b2
        embedding, norms = self._normalize_rows(raw)
        cache = (x_std, hidden, raw, norms)
        return embedding, cache

    @staticmethod
    def _backprop_normalization(
        embedding: np.ndarray,
        norms: np.ndarray,
        grad_embedding: np.ndarray,
    ) -> np.ndarray:
        # d(r / ||r||) = (g - z * <g,z>) / ||r||
        projection = np.sum(grad_embedding * embedding, axis=1, keepdims=True)
        return (grad_embedding - embedding * projection) / norms

    def _backward(
        self,
        cache: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        embedding: np.ndarray,
        grad_embedding: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x_std, hidden, _raw, norms = cache
        grad_raw = self._backprop_normalization(
            embedding,
            norms,
            grad_embedding,
        )

        grad_w2 = hidden.T @ grad_raw
        grad_b2 = np.sum(grad_raw, axis=0)
        grad_hidden = grad_raw @ self.w2.T
        grad_hidden_pre = grad_hidden * (1.0 - hidden * hidden)
        grad_w1 = x_std.T @ grad_hidden_pre
        grad_b1 = np.sum(grad_hidden_pre, axis=0)
        return grad_w1, grad_b1, grad_w2, grad_b2

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @staticmethod
    def _build_label_index(labels: np.ndarray) -> dict[int, np.ndarray]:
        return {
            int(label): np.flatnonzero(labels == label)
            for label in np.unique(labels)
        }

    @staticmethod
    def _sample_pairs(
        rng: np.random.Generator,
        labels: np.ndarray,
        label_index: dict[int, np.ndarray],
        pair_count: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        classes = np.asarray(sorted(label_index), dtype=np.int64)
        positive_classes = np.asarray(
            [c for c in classes if label_index[int(c)].size >= 2],
            dtype=np.int64,
        )
        if positive_classes.size == 0 or classes.size < 2:
            raise ValueError("Metric training needs at least two classes with repeated samples.")

        half = max(1, pair_count // 2)
        first: list[int] = []
        second: list[int] = []
        same: list[float] = []

        for _ in range(half):
            cls = int(rng.choice(positive_classes))
            pair = rng.choice(label_index[cls], size=2, replace=False)
            first.append(int(pair[0]))
            second.append(int(pair[1]))
            same.append(1.0)

        for _ in range(pair_count - half):
            cls_a, cls_b = rng.choice(classes, size=2, replace=False)
            first.append(int(rng.choice(label_index[int(cls_a)])))
            second.append(int(rng.choice(label_index[int(cls_b)])))
            same.append(0.0)

        order = rng.permutation(len(first))
        return (
            np.asarray(first, dtype=np.int64)[order],
            np.asarray(second, dtype=np.int64)[order],
            np.asarray(same, dtype=np.float32)[order],
        )

    @staticmethod
    def _contrastive_pair_loss_and_grads(
        z1: np.ndarray,
        z2: np.ndarray,
        same: np.ndarray,
        negative_similarity_margin: float,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        similarities = np.sum(z1 * z2, axis=1)
        positive_mask = same > 0.5
        negative_mask = ~positive_mask

        loss_values = np.zeros_like(similarities, dtype=np.float32)
        grad_similarity = np.zeros_like(similarities, dtype=np.float32)

        # Same-class points should point in the same direction (cosine -> 1).
        loss_values[positive_mask] = 1.0 - similarities[positive_mask]
        grad_similarity[positive_mask] = -1.0

        # Different classes are only penalised when they remain too similar.
        hinge = similarities[negative_mask] - negative_similarity_margin
        active = hinge > 0.0
        negative_indices = np.flatnonzero(negative_mask)
        active_indices = negative_indices[active]
        if active_indices.size:
            active_hinge = similarities[active_indices] - negative_similarity_margin
            loss_values[active_indices] = active_hinge * active_hinge
            grad_similarity[active_indices] = 2.0 * active_hinge

        batch_size = max(1, z1.shape[0])
        grad_similarity /= float(batch_size)
        grad_z1 = grad_similarity[:, None] * z2
        grad_z2 = grad_similarity[:, None] * z1
        return float(np.mean(loss_values)), grad_z1, grad_z2

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        epochs: int = 350,
        pair_count: int = 320,
        learning_rate: float = 0.006,
        weight_decay: float = 2e-4,
        negative_similarity_margin: float = 0.10,
        augmentation_noise: float = 0.015,
        patience: int = 45,
    ) -> MetricTrainingReport:
        x = np.asarray(features, dtype=np.float32)
        y = np.asarray(labels, dtype=np.int64).reshape(-1)
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected training features shaped (N, {self.input_dim}), got {x.shape}."
            )
        if x.shape[0] != y.shape[0]:
            raise ValueError("Feature and label counts do not match.")
        if np.unique(y).size < 2:
            raise ValueError("Metric training requires at least two classes.")

        self.input_mean = np.mean(x, axis=0).astype(np.float32)
        std = np.std(x, axis=0).astype(np.float32)
        self.input_std = np.maximum(std, 1e-3).astype(np.float32)
        x_std = self._standardize(x).astype(np.float32)

        rng = np.random.default_rng(self.seed)
        label_index = self._build_label_index(y)

        parameters = [self.w1, self.b1, self.w2, self.b2]
        adam_m = [np.zeros_like(parameter) for parameter in parameters]
        adam_v = [np.zeros_like(parameter) for parameter in parameters]
        beta1, beta2 = 0.9, 0.999
        adam_eps = 1e-8

        best_loss = float("inf")
        best_parameters = [parameter.copy() for parameter in parameters]
        stale_epochs = 0
        final_loss = float("inf")
        epochs_ran = 0

        for epoch in range(1, max(1, int(epochs)) + 1):
            idx1, idx2, same = self._sample_pairs(
                rng,
                y,
                label_index,
                max(16, int(pair_count)),
            )
            batch1 = x_std[idx1].copy()
            batch2 = x_std[idx2].copy()
            if augmentation_noise > 0.0:
                batch1 += rng.normal(0.0, augmentation_noise, batch1.shape).astype(np.float32)
                batch2 += rng.normal(0.0, augmentation_noise, batch2.shape).astype(np.float32)

            z1, cache1 = self._forward_standardized(batch1)
            z2, cache2 = self._forward_standardized(batch2)
            loss, grad_z1, grad_z2 = self._contrastive_pair_loss_and_grads(
                z1,
                z2,
                same,
                negative_similarity_margin,
            )

            grads1 = self._backward(cache1, z1, grad_z1)
            grads2 = self._backward(cache2, z2, grad_z2)
            gradients = [a + b for a, b in zip(grads1, grads2)]

            # L2 regularisation only on matrices, not biases.
            gradients[0] += weight_decay * self.w1
            gradients[2] += weight_decay * self.w2
            regularization = 0.5 * weight_decay * (
                float(np.sum(self.w1 * self.w1))
                + float(np.sum(self.w2 * self.w2))
            )
            final_loss = loss + regularization

            for index, (parameter, gradient) in enumerate(zip(parameters, gradients)):
                # Avoid a single difficult pair batch destabilising the tiny model.
                gradient = np.clip(gradient, -2.0, 2.0)
                adam_m[index] = beta1 * adam_m[index] + (1.0 - beta1) * gradient
                adam_v[index] = beta2 * adam_v[index] + (1.0 - beta2) * (gradient * gradient)
                m_hat = adam_m[index] / (1.0 - beta1**epoch)
                v_hat = adam_v[index] / (1.0 - beta2**epoch)
                parameter -= learning_rate * m_hat / (np.sqrt(v_hat) + adam_eps)

            epochs_ran = epoch
            if final_loss < best_loss - 1e-5:
                best_loss = final_loss
                best_parameters = [parameter.copy() for parameter in parameters]
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= max(5, int(patience)):
                    break

        self.w1[:], self.b1[:], self.w2[:], self.b2[:] = best_parameters
        self.trained = True

        embeddings = self.encode_batch(x)
        report = self._evaluate_training_geometry(embeddings, y)
        return MetricTrainingReport(
            input_dimension=self.input_dim,
            class_count=int(np.unique(y).size),
            sample_count=int(x.shape[0]),
            hidden_dimension=self.hidden_dim,
            embedding_dimension=self.embedding_dim,
            epochs_ran=epochs_ran,
            final_loss=float(best_loss),
            leave_one_out_accuracy=report[0],
            mean_same_distance=report[1],
            mean_different_distance=report[2],
        )

    @staticmethod
    def _evaluate_training_geometry(
        embeddings: np.ndarray,
        labels: np.ndarray,
    ) -> tuple[float, float, float]:
        classes = np.unique(labels)
        correct = 0
        evaluated = 0
        same_distances: list[float] = []
        different_distances: list[float] = []

        for i in range(embeddings.shape[0]):
            candidates: list[tuple[float, int]] = []
            for cls in classes:
                indices = np.flatnonzero(labels == cls)
                if cls == labels[i]:
                    indices = indices[indices != i]
                    if indices.size == 0:
                        continue
                prototype = np.mean(embeddings[indices], axis=0)
                prototype /= max(float(np.linalg.norm(prototype)), _EPS)
                distance = float(np.sqrt(np.mean((embeddings[i] - prototype) ** 2)))
                candidates.append((distance, int(cls)))

            if candidates:
                predicted = min(candidates, key=lambda item: item[0])[1]
                correct += int(predicted == int(labels[i]))
                evaluated += 1

            same_indices = np.flatnonzero(labels == labels[i])
            same_indices = same_indices[same_indices != i]
            if same_indices.size:
                same_distances.append(
                    float(
                        np.mean(
                            np.sqrt(
                                np.mean(
                                    (embeddings[same_indices] - embeddings[i]) ** 2,
                                    axis=1,
                                )
                            )
                        )
                    )
                )
            different_indices = np.flatnonzero(labels != labels[i])
            if different_indices.size:
                different_distances.append(
                    float(
                        np.mean(
                            np.sqrt(
                                np.mean(
                                    (embeddings[different_indices] - embeddings[i]) ** 2,
                                    axis=1,
                                )
                            )
                        )
                    )
                )

        accuracy = correct / max(evaluated, 1)
        mean_same = float(np.mean(same_distances)) if same_distances else 0.0
        mean_different = (
            float(np.mean(different_distances)) if different_distances else 0.0
        )
        return float(accuracy), mean_same, mean_different

    # ------------------------------------------------------------------
    # Inference / serialisation
    # ------------------------------------------------------------------

    def encode_batch(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected features shaped (N, {self.input_dim}), got {x.shape}."
            )
        x_std = self._standardize(x).astype(np.float32)
        embedding, _ = self._forward_standardized(x_std)
        return embedding.astype(np.float32)

    def encode(self, feature: np.ndarray) -> np.ndarray:
        return self.encode_batch(np.asarray(feature, dtype=np.float32))[0]

    def state_dict(self, prefix: str) -> dict[str, np.ndarray]:
        return {
            f"{prefix}_meta": np.asarray(
                [
                    self.FORMAT_VERSION,
                    self.input_dim,
                    self.hidden_dim,
                    self.embedding_dim,
                    self.seed,
                    int(self.trained),
                ],
                dtype=np.int64,
            ),
            f"{prefix}_mean": self.input_mean.astype(np.float32),
            f"{prefix}_std": self.input_std.astype(np.float32),
            f"{prefix}_w1": self.w1.astype(np.float32),
            f"{prefix}_b1": self.b1.astype(np.float32),
            f"{prefix}_w2": self.w2.astype(np.float32),
            f"{prefix}_b2": self.b2.astype(np.float32),
        }

    @classmethod
    def from_state_dict(cls, arrays: dict[str, np.ndarray], prefix: str) -> "NumpyMetricMLP":
        meta = np.asarray(arrays[f"{prefix}_meta"], dtype=np.int64)
        version, input_dim, hidden_dim, embedding_dim, seed, trained = meta.tolist()
        if version != cls.FORMAT_VERSION:
            raise ValueError(f"Unsupported metric encoder format: {version}")
        encoder = cls(input_dim, hidden_dim, embedding_dim, seed)
        encoder.input_mean = np.asarray(arrays[f"{prefix}_mean"], dtype=np.float32)
        encoder.input_std = np.asarray(arrays[f"{prefix}_std"], dtype=np.float32)
        encoder.w1 = np.asarray(arrays[f"{prefix}_w1"], dtype=np.float32)
        encoder.b1 = np.asarray(arrays[f"{prefix}_b1"], dtype=np.float32)
        encoder.w2 = np.asarray(arrays[f"{prefix}_w2"], dtype=np.float32)
        encoder.b2 = np.asarray(arrays[f"{prefix}_b2"], dtype=np.float32)
        encoder.trained = bool(trained)
        return encoder


class MetricEmbeddingBank:
    """Holds one frozen encoder per raw feature dimensionality (83/169 today)."""

    FORMAT_VERSION = 1

    def __init__(self, embedding_dim: int = 48) -> None:
        self.embedding_dim = int(embedding_dim)
        self.encoders: dict[int, NumpyMetricMLP] = {}
        self.reports: dict[int, MetricTrainingReport] = {}

    @property
    def active_dimensions(self) -> tuple[int, ...]:
        return tuple(sorted(self.encoders))

    @property
    def is_active(self) -> bool:
        return bool(self.encoders)

    def can_encode_dimension(self, dimension: int) -> bool:
        return int(dimension) in self.encoders

    def encode(self, feature: np.ndarray) -> np.ndarray:
        vector = np.asarray(feature, dtype=np.float32).reshape(-1)
        encoder = self.encoders.get(int(vector.shape[0]))
        if encoder is None:
            # Deliberate fallback for a hand configuration with no learned source
            # metric yet. Keeping the raw vector usable is safer than inventing a
            # random embedding. Such samples remain isolated by feature dimension.
            return vector.copy()
        return encoder.encode(vector)

    def fit_from_labeled_groups(
        self,
        groups: dict[str, Iterable[np.ndarray]],
        *,
        hidden_dim: int = 96,
        min_classes: int = 2,
        min_samples_per_class: int = 2,
        seed: int = 42,
    ) -> list[MetricTrainingReport]:
        by_dimension: dict[int, list[tuple[str, np.ndarray]]] = {}
        for label, samples in groups.items():
            prepared = [np.asarray(sample, dtype=np.float32).reshape(-1) for sample in samples]
            if not prepared:
                continue
            dimensions = {sample.shape[0] for sample in prepared}
            if len(dimensions) != 1:
                raise ValueError(f"Gesture '{label}' mixes feature dimensions.")
            if len(prepared) < min_samples_per_class:
                continue
            dimension = int(prepared[0].shape[0])
            by_dimension.setdefault(dimension, []).extend((label, sample) for sample in prepared)

        reports: list[MetricTrainingReport] = []
        for dimension, records in sorted(by_dimension.items()):
            labels_in_dimension = sorted({label for label, _ in records})
            valid_labels = [
                label
                for label in labels_in_dimension
                if sum(1 for item_label, _ in records if item_label == label)
                >= min_samples_per_class
            ]
            if len(valid_labels) < min_classes:
                continue

            label_to_id = {label: index for index, label in enumerate(valid_labels)}
            filtered = [(label, sample) for label, sample in records if label in label_to_id]
            x = np.stack([sample for _, sample in filtered]).astype(np.float32)
            y = np.asarray([label_to_id[label] for label, _ in filtered], dtype=np.int64)

            encoder = NumpyMetricMLP(
                input_dim=dimension,
                hidden_dim=hidden_dim,
                embedding_dim=self.embedding_dim,
                seed=seed + dimension,
            )
            report = encoder.fit(x, y)
            self.encoders[dimension] = encoder
            self.reports[dimension] = report
            reports.append(report)

        return reports

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {
            "bank_meta": np.asarray(
                [self.FORMAT_VERSION, self.embedding_dim, len(self.encoders)],
                dtype=np.int64,
            ),
            "dimensions": np.asarray(sorted(self.encoders), dtype=np.int64),
        }
        for dimension, encoder in self.encoders.items():
            arrays.update(encoder.state_dict(f"d{dimension}"))
        temporary = destination.with_suffix(destination.suffix + ".tmp.npz")
        np.savez_compressed(temporary, **arrays)
        temporary.replace(destination)

    @classmethod
    def load(cls, path: str | Path) -> "MetricEmbeddingBank":
        source = Path(path)
        with np.load(source, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        meta = np.asarray(arrays["bank_meta"], dtype=np.int64)
        version, embedding_dim, _count = meta.tolist()
        if version != cls.FORMAT_VERSION:
            raise ValueError(f"Unsupported metric bank format: {version}")
        bank = cls(embedding_dim=int(embedding_dim))
        for dimension in np.asarray(arrays["dimensions"], dtype=np.int64).tolist():
            bank.encoders[int(dimension)] = NumpyMetricMLP.from_state_dict(
                arrays,
                f"d{int(dimension)}",
            )
        return bank

    @classmethod
    def train_from_learner(
        cls,
        learner,
        *,
        embedding_dim: int = 48,
        hidden_dim: int = 96,
        seed: int = 42,
    ) -> tuple["MetricEmbeddingBank", list[MetricTrainingReport]]:
        groups = {
            name: [sample.copy() for sample in gesture.samples]
            for name, gesture in learner.gestures.items()
        }
        bank = cls(embedding_dim=embedding_dim)
        reports = bank.fit_from_labeled_groups(
            groups,
            hidden_dim=hidden_dim,
            seed=seed,
        )
        return bank, reports
