# V3.2 Research Track — EVT-Based Open-Set Recognition

## Goal

V3.2 changes only the **static UNKNOWN / open-set decision**. The V3.1 geometry-aware hybrid descriptor remains unchanged, as do Smart Capture, multi-prototype positive memory, feedback, persistence, dynamic DTW recognition, and prediction stabilization.

The motivation is to replace the primary hand-tuned radius rejection rule with a boundary model that uses the **distribution of distances from positive exemplars to negative evidence**.

## Research inspiration

The design is inspired by the core probability-of-sample-inclusion formulation from:

E. M. Rudd, L. P. Jain, W. J. Scheirer, and T. E. Boult, **The Extreme Value Machine**, IEEE TPAMI, 2018.

For a positive exemplar `x_i`, the EVM considers margin estimates based on half-distances to nearby samples from other classes:

`m_ij = ||x_i - x_j|| / 2`

where `x_j` is a negative sample. Extreme Value Theory motivates fitting a Weibull distribution to the smallest observed margin distances. A radial inclusion score for a query `x` is then:

`Psi_i(x) = exp( - ( ||x_i - x|| / lambda_i ) ^ kappa_i )`

where `kappa_i` and `lambda_i` are the fitted Weibull shape and scale parameters.

## What this project implements

The implementation preserves the project's online few-shot architecture:

1. A user teaches a gesture from a few live demonstrations.
2. V3.1 converts each static pose into the geometry-aware hybrid feature vector.
3. When at least one competing class (or sufficient hard-negative feedback) exists, each positive exemplar receives a Weibull radial inclusion model.
4. Negative evidence consists of compatible samples from other learned gesture classes plus explicit hard-negative user corrections for that class.
5. At inference, every compatible class receives a probability-of-inclusion style score.
6. The class with the highest inclusion is accepted only when it exceeds the open-set threshold; otherwise the result is `UNKNOWN`.
7. When insufficient negative evidence exists, the system temporarily falls back to the previously validated V3.1 local-radius rejection mechanism.

## Important distinction from a canonical EVM implementation

V3.2 adopts the EVM's **margin-tail Weibull fitting and radial inclusion function**, but it does not yet implement the canonical EVM's extreme-vector set-cover model reduction or large-scale partial model fitting. All few-shot exemplars are retained at this stage so V3.2 isolates open-set boundary modeling from the later V3.4 exemplar-memory experiment.

This distinction should be maintained in publications: the current implementation is best described as an **EVM-inspired EVT open-set gate for an online few-shot prototype learner**, not as a full reproduction of the canonical EVM system.

## Incremental behavior

Adding a new gesture changes the negative margin evidence available to existing classes. V3.2 therefore rebuilds the small EVT boundary cache whenever positive class memory changes. Since gesture memories are deliberately bounded and few-shot, this operation is inexpensive and does not constitute global neural-network retraining.

Hard-negative feedback is used twice:

- as explicit negative margin evidence during future Weibull fitting;
- as an immediate veto for a pose the user has explicitly rejected.

## No new dependency

The two-parameter zero-location Weibull fit is implemented directly using the maximum-likelihood score equation with numerically robust bisection. SciPy is not required.

## Research parameters

Current experimental defaults:

- tail size: 10 nearest negative margin estimates;
- minimum negative evidence: 3 samples;
- inclusion threshold: 0.35;
- class score: maximum exemplar inclusion (`top-k = 1`).

These values are **research defaults, not final optimized values**. The EVM paper recommends choosing the inclusion threshold using non-test validation data. They should therefore be tuned only through a pre-declared validation protocol before publication claims are made.

## Recommended V3.2 experiments

1. V3.1 radius rejection vs V3.2 EVT rejection with the same hybrid features.
2. Known-gesture true acceptance rate.
3. Unknown-gesture true rejection rate.
4. AUROC / AUPR using the EVT inclusion score.
5. FPR at selected known-gesture TPR operating points.
6. Effect of tail size (`tau`).
7. Effect of inclusion threshold (`delta`).
8. Number of known classes and degree of openness.
9. Before/after hard-negative feedback.
10. Cross-user unknown rejection.

The key experimental rule is to keep the V3.1 feature representation fixed while comparing only the open-set decision mechanism.
