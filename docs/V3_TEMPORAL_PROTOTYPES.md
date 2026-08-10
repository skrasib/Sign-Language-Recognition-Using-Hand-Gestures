# V3.5 — DTW-Aligned Temporal Prototypes

## Goal

V3.5 changes the representation used for **dynamic gesture classes**.

Before V3.5, a dynamic class stored a few normalized landmark trajectories and
recognized a new movement using the nearest individual demonstration under
multivariate Dynamic Time Warping (DTW).

V3.5 keeps those demonstrations as the source memory, but derives one or two
**temporal prototypes** from them. Runtime recognition compares a completed live
trajectory with the temporal prototype(s) instead of comparing it with every stored
demonstration.

No camera images or video frames are stored.

## Why a normal arithmetic average is not enough

Two people can perform the same movement with slightly different timing. For example,
the midpoint of a swipe can occur at frame 18 in one demonstration and frame 23 in
another. Averaging frame 18 with frame 18 can therefore blur the movement.

DTW explicitly aligns sequences in time before measuring their discrepancy. The
prototype builder uses that alignment to average **corresponding movement phases**
rather than blindly averaging equal frame indices.

## V3.5 prototype construction

For each dynamic gesture:

1. Choose a real demonstration near the centre of the demonstrations (a DTW medoid).
2. Align every demonstration to the current centre with the project's multivariate
   DTW cost.
3. For each centre time step, collect the shape and motion observations aligned to
   that step.
4. Average those aligned observations.
5. Recompute velocity from the averaged motion trajectory.
6. Repeat the alignment/update procedure for a small number of iterations.

This is a lightweight **DTW-barycenter / DBA-style** temporal averaging procedure.
The core idea is based on the DTW Barycenter Averaging literature:

- Petitjean, Ketterlin & Gancarski, *A global averaging method for dynamic time
  warping, with applications to clustering*, Pattern Recognition, 2011.
  DOI: 10.1016/j.patcog.2010.09.013

The project deliberately uses the existing weighted multivariate gesture DTW cost:

- normalized hand-shape sequence;
- normalized wrist/hand motion sequence;
- velocity sequence.

## Relationship to Soft-DTW

Cuturi & Blondel's Soft-DTW replaces DTW's hard minimum over alignments with a smooth
minimum, making the objective differentiable and suitable for gradient-based time
series averaging and learning:

- Cuturi & Blondel, *Soft-DTW: a Differentiable Loss Function for Time-Series*,
  ICML 2017.

Soft-DTW is highly relevant to this research direction, but **V3.5 is not a
Soft-DTW implementation**. The current application does not need gradients, and the
DBA-style update integrates directly with the existing NumPy-only multivariate DTW
engine without adding a compiled/scientific optimization dependency.

A later controlled experiment can compare:

- nearest-template DTW;
- DTW barycenter prototypes (V3.5);
- Soft-DTW/Soft-DTW-divergence barycenters.

## One or two prototypes

With the default three teaching demonstrations, V3.5 builds **one** temporal
prototype.

If a class later has at least five retained demonstrations and they form two
supported temporal modes, V3.5 can derive up to two prototypes. It starts from the
farthest pair of demonstrations, partitions demonstrations by DTW proximity, and
only keeps the split when both groups contain at least two demonstrations. This
prevents a single unusual performance from automatically becoming its own mode.

## Recognition after V3.5

Before:

```text
new movement
    ↓
DTW vs demo 1
DTW vs demo 2
DTW vs demo 3
    ↓
nearest demonstration
```

V3.5:

```text
three live demonstrations
    ↓
DTW alignment + temporal averaging
    ↓
temporal prototype

new movement
    ↓
DTW vs prototype
    ↓
class score / threshold / UNKNOWN
```

The class threshold is rebuilt around the temporal prototype(s). To avoid an
artificially tiny threshold from only three very similar demonstrations, V3.5 uses
both demonstration-to-prototype distances and half of the previous pairwise class
spread estimate, then keeps the existing minimum threshold floor.

## Persistence

The JSON persistence format does not change.

Only the original normalized landmark trajectories are saved. Temporal prototypes
are **derived state** and are rebuilt when the dynamic memory is loaded. This keeps
existing V3 dynamic memory compatible and avoids unnecessary duplicated data.

## Ablation support

The old nearest-template behavior remains available:

```python
DynamicGestureLearner(temporal_prototype_strategy="templates")
```

V3.5 default:

```python
DynamicGestureLearner(
    temporal_prototype_strategy="dtw_barycenter",
    max_temporal_prototypes=2,
    prototype_iterations=4,
)
```

This allows a controlled future comparison with exactly the same demonstrations,
segmentation, DTW cost, and thresholds.

## What V3.5 does not claim

- It is not a reproduction of Soft-DTW.
- It is not a full reproduction of every detail of the original DBA algorithm.
- It does not yet prove that temporal prototypes improve accuracy; that requires a
  held-out evaluation.
- It does not change static V3.1–V3.4 recognition.
- It does not change the MediaPipe tracking API; that remains V3.6.

## Recommended evaluation

For each dynamic class, compare nearest-template DTW against V3.5 temporal
prototypes under the same captured demonstrations:

1. recognition accuracy/F1;
2. unseen performance speed (slow vs fast execution);
3. temporal warping / uneven-speed execution;
4. between-user movement variation;
5. UNKNOWN rejection for unrelated motion;
6. mean recognition latency;
7. number of DTW comparisons per completed gesture.

The expected computational advantage is strongest as the retained demonstration
count grows: one or two prototypes require fewer DTW comparisons than matching every
stored demonstration.
