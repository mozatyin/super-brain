# Hybrid Rule-Based + LLM Detection Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Split 70 personality traits into rule-based (~7 deterministic) and LLM-semantic (~63), reducing variance and cost for measurable traits.

**Architecture:** Add `compute_direct_scores()` to behavioral_features.py. Detector excludes rule-based traits from LLM batches, merges results, applies post-processing selectively.

**Tech Stack:** Python 3.12, existing behavioral_features.py + detector.py

---

## Context

Super Brain currently sends all 70 traits to LLM in 7 batches. Some traits (verbosity, politeness) are directly measurable from text statistics — the system prompt already tells the LLM "count words, use this scale." This wastes tokens and introduces unnecessary LLM variance.

## Decision: Approach A — Extend behavioral_features.py

Add `compute_direct_scores(bf: BehavioralFeatures) -> dict[str, float]` alongside existing `compute_adjustments()`. Detector calls it before LLM loop, excludes those traits from batches, merges at the end.

## Rule-Based Trait List (7 traits)

| Trait | Primary Signal | Why Rule-Based Suffices |
|---|---|---|
| verbosity | avg_words_per_turn | Direct measurement, definition = talk length |
| politeness | politeness_ratio | Direct count of please/thanks/sorry |
| decisiveness | hedging - absolutist + decisiveness ratios | Hesitation vs certainty word balance |
| curiosity | question_ratio + curiosity_ratio | Question frequency + curiosity phrases |
| hot_cold_oscillation | words_std | Response length variance = engagement oscillation |
| self_mythologizing | self_ref_ratio (primary) | Self-reference density as proxy |
| optimism | pos/(pos+neg) emotion ratio | Positive-to-negative framing ratio |

Traits NOT included (semantic understanding required): trust, depression, narcissism, social_dominance, charm_influence, fantasy, humor styles, dark traits, attachment styles.

## Mapping: Piecewise Linear Interpolation

Each trait uses `_interpolate(value, breakpoints, scores)` for continuous, smooth scoring:

```python
def _interpolate(value, breakpoints, scores):
    if value <= breakpoints[0]: return scores[0]
    if value >= breakpoints[-1]: return scores[-1]
    for i in range(len(breakpoints) - 1):
        if value <= breakpoints[i + 1]:
            t = (value - breakpoints[i]) / (breakpoints[i + 1] - breakpoints[i])
            return scores[i] + t * (scores[i + 1] - scores[i])
```

### Mapping Tables

| Trait | Breakpoints | Scores |
|---|---|---|
| verbosity | [0, 40, 80, 150, 300] | [0.10, 0.20, 0.42, 0.58, 0.80] |
| politeness | [0, 0.005, 0.015, 0.030, 0.060] | [0.12, 0.25, 0.38, 0.58, 0.75] |
| decisiveness | [-0.03, -0.01, 0, 0.01, 0.03] | [0.25, 0.38, 0.50, 0.62, 0.75] |
| curiosity | [0, 0.05, 0.15, 0.25, 0.40] | [0.28, 0.38, 0.50, 0.62, 0.72] |
| hot_cold_oscillation | [0, 30, 60, 100, 160] | [0.20, 0.28, 0.38, 0.55, 0.70] |
| self_mythologizing | [0, 0.03, 0.06, 0.09, 0.14] | [0.25, 0.32, 0.40, 0.52, 0.65] |
| optimism | [0, 0.3, 0.5, 0.7, 1.0] | [0.22, 0.35, 0.47, 0.60, 0.72] |

Breakpoints derived from: system prompt calibration ranges, existing adjustment rules, empirical trait distributions from 9-character literary evaluation.

## Data Flow in analyze()

```
1. bf = extract_features(conversation)
2. direct_scores = compute_direct_scores(bf)           # 7 traits, deterministic
3. for batch in DIMENSION_BATCHES:
     batch_traits = [t for t if t.name not in direct_scores]
     if empty: skip
     llm_result = call_llm(batch_traits)               # ~63 traits
4. all_traits = rule_based (conf=0.95) + llm_traits
5. compute_adjustments(bf) → apply ONLY to llm_traits   # no double-count
6. _validate_consistency(all_traits)                     # applies to ALL
7. _calibrate_known_biases(all_traits)                   # skip rule-based
8. _bayesian_shrinkage(all_traits)                       # rule-based conf=0.95, no shrink
```

### Post-Processing Rules

- **Consistency validation**: Applies to all traits (rule-based can contradict LLM traits)
- **Calibration corrections**: Skip rule-based traits (corrections are for LLM bias)
- **Bayesian shrinkage**: Rule-based confidence=0.95 > threshold 0.60, so no shrinkage applied
- **Post-hoc adjustments**: Skip rule-based traits (already computed from same features)

## Validation Plan (must pass before commit)

**Layer 1: Unit tests (zero cost)**
- 256 existing tests pass
- 21 new tests: 7 traits x 3 cases (low/mid/high)
- Edge cases: empty conversation, 1 message, extreme values
- Interpolation continuity: adjacent inputs produce smooth output

**Layer 2: Integration tests (zero cost)**
- Rule-based traits excluded from LLM prompt (mock verification)
- Rule-based traits skip calibration and adjustment
- Consistency rules still apply to rule-based traits

**Layer 3: Full A/B comparison (API cost)**
- Run full_comparison.py on 9 literary characters
- Compare against gt_v41.json (old pipeline GT)
- Pass criteria:
  - Overall MAE ≤ 0.085 (current: 0.081, max regression: 0.004)
  - All 13 dimensions MAE < 0.16 (no new warnings)
  - Rule-based trait variance = 0 (deterministic)
- Fail → adjust breakpoints/scores, re-run. Do NOT commit until pass.
