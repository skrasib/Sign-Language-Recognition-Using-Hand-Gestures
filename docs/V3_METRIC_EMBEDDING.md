# V3.3 — Frozen Learned Metric Embedding

## Goal

V3.3 inserts a learned feature-mapping stage between the V3 hybrid hand descriptor and the existing online prototype/EVT learner.

The runtime path becomes:

```text
Camera
  -> MediaPipe landmarks
  -> V3 hybrid descriptor (83-D one hand / 169-D two hands)
  -> frozen metric MLP (48-D when available)
  -> online exemplars + adaptive prototypes
  -> EVT open-set rejection
  -> stabilized prediction
```

The user can still add a new runtime gesture immediately. The class-specific memory is still exemplar/prototype based; the metric encoder is not retrained when a normal runtime class is added.

## Research inspiration

This stage is inspired by metric-learning approaches such as Prototypical Networks and by the 2026 CVPR Workshop paper *Geometry-Aware Metric Learning for Cross-Lingual Few-Shot Sign Language Recognition on Static Hand Keypoints*.

The CVPRW work uses MediaPipe landmarks, geometry-aware angle descriptors, a lightweight MLP encoder and prototype classification. Its public implementation uses a 2-hidden-layer MLP (256, 256) with a 128-D embedding for the reported experiments.

V3.3 does **not** claim to reproduce that training protocol. The current implementation deliberately uses a much smaller NumPy MLP (96 hidden units -> 48-D embedding by default) because the local live-taught source memory is tiny compared with a public pretraining corpus.

## What is learned?

The encoder receives a hybrid feature vector and learns a new metric space using labeled source gestures.

Training uses balanced positive/negative pairs:

- two samples from the same gesture are pulled closer;
- two samples from different gestures are pushed apart when they are still too similar;
- output embeddings are L2-normalized.

The encoder is saved to:

```text
data/v3/metric_encoder_v33.npz
```

No image or video data is saved.

## Bootstrap source

For convenience, first V3.3 launch looks for the existing raw hybrid V3.1/V3.2 memory:

```text
data/v3/gesture_memory_hybrid.json
```

If no metric encoder exists and there are at least two compatible gesture classes with repeated samples, V3.3 trains the encoder once and freezes it.

The V3.3 runtime gesture memory is separate:

```text
data/v3/gesture_memory_metric_v33.json
```

On first launch, existing V3.2 classes are embedded and migrated into this new runtime memory so V3.2 and V3.3 can be compared without destroying the raw source memory.

## Smart Capture remains in raw geometry space

Smart Capture intentionally measures stability and diversity using the raw hybrid descriptor rather than the learned embedding.

Reason: a good metric encoder may compress natural same-class variation very strongly. If duplicate filtering were performed only after the metric encoder, useful live examples could be discarded as near-identical before they are learned.

Only after Smart Capture chooses useful raw samples are those samples mapped to the metric space for prototype learning.

## Manual encoder training

The app trains automatically only when no encoder exists. For controlled experiments, explicitly train/retrain with:

```powershell
uv run python scripts/train_v3_metric_encoder.py
```

A different raw source memory can be supplied:

```powershell
uv run python scripts/train_v3_metric_encoder.py --source path\to\source_memory.json
```

If the encoder is deliberately retrained, delete `data/v3/gesture_memory_metric_v33.json` before the next run so runtime classes are rebuilt using the new embedding.

## Important publication caveat

Training the metric encoder on the same gesture classes that are later used for evaluation is **not** a valid unseen-class few-shot experiment.

The local V3.2-memory bootstrap is an engineering/personalization mode that lets the learned-metric pipeline be tested immediately.

For formal experiments, use disjoint data, for example:

```text
Source users/classes -> train frozen encoder
Target users/classes -> few-shot runtime teaching only
Held-out target trials -> evaluation
```

A future publication-quality experiment should compare at least:

1. V2 coordinate descriptor;
2. V3.1 hybrid descriptor;
3. V3.2 hybrid + EVT;
4. V3.3 hybrid + frozen learned metric + EVT.

The encoder source data must be disjoint from the target evaluation conditions for claims about generalization.

## What V3.3 does not change

- MediaPipe tracking remains unchanged.
- V3.1 geometry extraction remains unchanged.
- V3.2 EVT open-set rejection remains active after the metric mapping.
- hard-negative feedback remains active;
- multi-prototype learning remains active;
- static runtime class addition remains immediate;
- dynamic DTW recognition is deliberately unchanged until the planned temporal-prototype stage.

## References

- Snell, Swersky, Zemel. *Prototypical Networks for Few-shot Learning*. NeurIPS 2017.
- Chamachot, Lertniponphan. *Geometry-Aware Metric Learning for Cross-Lingual Few-Shot Sign Language Recognition on Static Hand Keypoints*. CVPR Workshops 2026.
