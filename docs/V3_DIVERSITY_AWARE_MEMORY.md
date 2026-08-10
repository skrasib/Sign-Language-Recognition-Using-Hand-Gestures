# V3.4 — Diversity-Aware Exemplar Memory

## Goal

V3.4 changes **which numerical gesture examples are retained after a bounded class
memory becomes full**. It does not store camera images and it does not retrain the
V3.3 metric encoder during normal live teaching.

Before V3.4, positive feedback memory was effectively recency based: when a class
exceeded its sample budget, the oldest example was discarded. Hard-negative memory
used the same FIFO idea. That is simple, but a recent streak of very similar poses
can gradually replace older examples that represented useful natural variations of
the same gesture.

V3.4 replaces this with a deterministic geometry-based memory policy.

## Positive exemplar memory

For a class with more examples than its memory budget:

1. Compute pairwise distances in the **current recognition space**. In V3.3 this is
   normally the frozen 48-D metric embedding.
2. Keep the class **medoid** first. The medoid is a real observed sample near the
   centre of the class.
3. Repeatedly add the sample that is farthest from its nearest already-selected
   exemplar.
4. Stop at the fixed memory budget.

This is a lightweight **k-center-style / core-set-inspired farthest-first** policy.
Its purpose is coverage: retain central behaviour plus naturally different valid
variations instead of retaining examples only because they arrived recently.

This direction is inspired by the core-set view of subset selection in Sener and
Savarese, *Active Learning for Convolutional Neural Networks: A Core-Set Approach*,
ICLR 2018. Their complete active-learning method is broader than the small bounded
memory routine implemented here.

Reference: https://arxiv.org/abs/1708.00489

## Hard-negative memory

Hard negatives are user-corrected examples that were close enough to be confused
with a known gesture. They are especially important to the V3.2 EVT open-set
boundary.

V3.4 therefore does not treat every old negative equally:

1. Rank hard negatives by distance to the nearest positive exemplar.
2. Protect roughly half of the memory for the closest (hardest) negatives.
3. Build a candidate pool of boundary-relevant negatives.
4. Fill the remaining memory with farthest-first diverse negatives from that pool.

The result intentionally balances **boundary relevance** and **diversity**.

## Relationship to incremental-learning literature

Exemplar selection is a recurring idea in class-incremental learning. iCaRL, for
example, maintains a bounded exemplar set and uses an exemplar-selection procedure
rather than storing all historical training data.

Reference: Rebuffi et al., *iCaRL: Incremental Classifier and Representation
Learning*, CVPR 2017.
https://openaccess.thecvf.com/content_cvpr_2017/html/Rebuffi_iCaRL_Incremental_Classifier_CVPR_2017_paper.html

V3.4 should be described as **inspired by bounded exemplar/core-set selection**, not
as a reproduction of iCaRL. We do not perform iCaRL's representation learning,
distillation, or nearest-mean class-incremental training procedure.

## What changes in the app?

The normal user workflow is unchanged:

- teach a gesture live;
- use Correct/Wrong feedback;
- add new classes without global retraining;
- persist numerical features locally.

The difference appears once a class accumulates more than its bounded memory. V3.4
selects which examples survive based on coverage rather than age.

Default policies in the V3 app:

- positive exemplars: `diversity`
- hard negatives: `boundary_diversity`

The old FIFO policy remains available in `OnlineGestureLearner` for controlled
ablation experiments:

```python
OnlineGestureLearner(
    exemplar_memory_strategy="fifo",
    hard_negative_memory_strategy="fifo",
)
```

This is important for later publication-oriented evaluation because V3.4 can be
compared directly against the previous recency baseline under the **same memory
budget**.

## What V3.4 does not claim

- It does not reproduce the complete k-center optimization from the ICLR core-set
  paper.
- It does not reproduce iCaRL.
- It does not prove that diversity memory improves recognition accuracy until a
  controlled evaluation is run.
- It does not alter the frozen V3.3 metric encoder.
- It does not alter the dynamic DTW gesture memory yet; dynamic temporal prototypes
  are a separate V3.5 milestone.

## Recommended V3.4 experiment

For each gesture, stream additional valid feedback samples until the bounded memory
is exercised, then compare:

1. FIFO memory vs diversity-aware memory;
2. identical memory budget (for example 20, 40, or 60 examples/class);
3. recognition on old pose variations and recent pose variations;
4. open-set rejection before/after accumulating hard negatives;
5. class coverage radius and recognition F1/accuracy.

A strong hypothesis is that diversity-aware memory will retain older meaningful
variations better under long-running personalization, particularly when recent
feedback is biased toward one pose/viewpoint.
