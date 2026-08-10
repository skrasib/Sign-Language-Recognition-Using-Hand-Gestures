# V3 Research Track — Geometry-Aware Hybrid Representation

## Scope

V3 is a separate research line. The existing V2 application remains available as the stable baseline. V3 starts by changing only the **static hand representation** while preserving the online few-shot learner, open-set logic, feedback loop, multi-prototype memory, persistence, stabilization, and dynamic DTW pipeline.

## Research inspiration

The initial representation is inspired by:

Chayanin Chamachot and Kanokphan Lertniponphan, **Geometry-Aware Metric Learning for Cross-Lingual Few-Shot Sign Language Recognition on Static Hand Keypoints**, CVPR Workshops 2026.

The paper evaluates three MediaPipe-keypoint representations: 63-D coordinates, 20-D inter-joint angles, and an 83-D coordinate+angle representation. V3 adopts the geometry idea, but integrates it into this project's live user-defined online-learning setting rather than reproducing the paper's cross-lingual episodic-learning experiment.

## V3 representations

The shared feature builder now supports:

- `coordinate`: V2-compatible 63-D one-hand / 129-D two-hand representation.
- `angle`: 20-D one-hand / 43-D two-hand representation.
- `hybrid`: 83-D one-hand / 169-D two-hand representation.

The V3 application uses `hybrid` for static gestures.

### Why hybrid?

Coordinates preserve useful pose/orientation information. Inter-joint angles describe hand shape and are invariant to global translation, uniform scale, and 3D rotation. The hybrid representation keeps both sources of information.

The current V3 hybrid descriptor standardizes the coordinate and angle blocks independently before concatenation. This avoids a trivial numerical-scale imbalance while leaving the online learner itself unchanged.

## What has deliberately NOT changed yet

- No pretrained/frozen MLP encoder has been added yet.
- V3.2 adds an EVM-inspired EVT open-set gate; see `docs/V3_OPEN_SET_EVT.md`.
- No exemplar-memory core-set selection has been added yet.
- Dynamic gestures still use the existing coordinate representation and DTW engine.
- MediaPipe tracking still uses the current tracker implementation.

These correspond to later V3 milestones and should be evaluated independently rather than bundled into one untestable change.

## Separate V3 memory

V3 uses its own local memory under `data/v3/`. Existing V2 memories are intentionally not migrated because 63-D/129-D V2 samples are not compatible with 83-D/169-D V3 samples.

This separation also makes V2 vs V3 ablation experiments reproducible.

## Recommended experiments

1. Coordinate vs angle vs hybrid within the same online learner.
2. Robustness to wrist rotation / camera viewpoint change.
3. Robustness to camera distance / hand size.
4. Number of useful Smart Capture samples required per representation.
5. Intra-class spread and local radius comparison.
6. Known-class accuracy and UNKNOWN rejection under controlled perturbations.
7. Cross-user evaluation once the protocol is finalized.
