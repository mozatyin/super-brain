# V2.8 ThinkSlow Trajectory Ensemble Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create an ensemble that blends the Detector's one-shot personality estimates with ThinkSlow's progressive trajectory of partial trait observations, using confidence-weighted averaging with ThinkSlow capped at 40% influence.

**Architecture:** A new `super_brain/ensemble.py` module provides `blend_with_trajectory()` which, for each detected trait, collects all ThinkSlow estimates and their confidence scores across cycles, computes a confidence-weighted mean, and blends it with the Detector's value proportionally to mean ThinkSlow confidence (capped at `max_ts_weight=0.4`). Traits not observed by ThinkSlow pass through unchanged. The function is wired into `detect_and_compare()` via a new `ts_results` parameter, and `run_eval()` passes the already-available `ts_results` list through.

**Tech Stack:** Python 3.12, Pydantic, pytest

**Dependency:** This plan assumes V2.6 (Soul-informed detection) has already been applied. If V2.6 is not yet applied, `detect_and_compare()` will have its original 4-parameter signature and Task 2 must add both `soul` and `ts_results` parameters. The code below accounts for V2.6 being applied first (i.e., `soul` parameter already exists).

---

### Task 1: Create ensemble module + tests

**Files:**
- Create: `/Users/michael/super-brain/super_brain/ensemble.py`
- Create: `/Users/michael/super-brain/tests/test_ensemble.py`

**Step 1: Write the failing test**

Create `/Users/michael/super-brain/tests/test_ensemble.py`:

```python
"""Tests for V2.8 ensemble blending."""

from super_brain.ensemble import blend_with_trajectory, _weighted_mean
from super_brain.models import (
    PersonalityDNA, Trait, Evidence, SampleSummary, ThinkSlowResult,
)


def _make_detected(traits_dict: dict[str, float]) -> PersonalityDNA:
    """Create a detected profile from {name: value} dict."""
    traits = [
        Trait(dimension="TEST", name=name, value=value, confidence=0.7,
              evidence=[Evidence(text="test", source="detector")])
        for name, value in traits_dict.items()
    ]
    return PersonalityDNA(
        id="detected",
        sample_summary=SampleSummary(
            total_tokens=100, conversation_count=1,
            date_range=["u", "u"], contexts=["eval"],
            confidence_overall=0.7,
        ),
        traits=traits,
    )


def _make_ts_result(traits_dict: dict[str, tuple[float, float]]) -> ThinkSlowResult:
    """Create ThinkSlowResult from {name: (value, confidence)} dict."""
    traits = [
        Trait(dimension="TEST", name=name, value=val, confidence=conf)
        for name, (val, conf) in traits_dict.items()
    ]
    conf_map = {name: conf for name, (val, conf) in traits_dict.items()}
    return ThinkSlowResult(
        partial_profile=PersonalityDNA(
            id="ts",
            sample_summary=SampleSummary(
                total_tokens=50, conversation_count=1,
                date_range=["u", "u"], contexts=["ts"],
                confidence_overall=0.5,
            ),
            traits=traits,
        ),
        confidence_map=conf_map,
        low_confidence_traits=[],
        observations=[],
        incisive_questions=[],
        info_staleness=0.5,
    )


def test_weighted_mean():
    assert abs(_weighted_mean([0.8, 0.2], [1.0, 1.0]) - 0.5) < 0.01
    assert abs(_weighted_mean([0.8, 0.2], [0.9, 0.1]) - 0.74) < 0.01


def test_weighted_mean_zero_weights():
    result = _weighted_mean([0.5, 0.6], [0.0, 0.0])
    assert abs(result - 0.55) < 0.01  # falls back to simple mean


def test_weighted_mean_empty():
    result = _weighted_mean([], [])
    assert result == 0.5  # fallback for empty input


def test_blend_no_ts_results():
    detected = _make_detected({"trust": 0.7, "anxiety": 0.3})
    result = blend_with_trajectory(detected, [])
    assert result.traits[0].value == 0.7
    assert result.traits[1].value == 0.3


def test_blend_with_single_ts():
    detected = _make_detected({"trust": 0.7})
    ts = [_make_ts_result({"trust": (0.5, 0.8)})]
    result = blend_with_trajectory(detected, ts)
    # ts_weight = 0.8 * 0.4 = 0.32, det_weight = 0.68
    # blended = 0.68 * 0.7 + 0.32 * 0.5 = 0.476 + 0.16 = 0.636
    assert abs(result.traits[0].value - 0.636) < 0.01


def test_blend_with_multiple_ts():
    detected = _make_detected({"trust": 0.7})
    ts = [
        _make_ts_result({"trust": (0.4, 0.6)}),
        _make_ts_result({"trust": (0.5, 0.8)}),
        _make_ts_result({"trust": (0.6, 0.7)}),
    ]
    result = blend_with_trajectory(detected, ts)
    # ts_avg = weighted_mean([0.4, 0.5, 0.6], [0.6, 0.8, 0.7])
    #        = (0.24 + 0.40 + 0.42) / (0.6 + 0.8 + 0.7)
    #        = 1.06 / 2.1 ≈ 0.5048
    # mean_conf = (0.6 + 0.8 + 0.7) / 3 ≈ 0.7
    # ts_weight = 0.7 * 0.4 = 0.28, det_weight = 0.72
    # blended = 0.72 * 0.7 + 0.28 * 0.5048 = 0.504 + 0.1413 ≈ 0.645
    assert abs(result.traits[0].value - 0.645) < 0.02


def test_blend_trait_not_in_ts():
    detected = _make_detected({"trust": 0.7, "anxiety": 0.3})
    ts = [_make_ts_result({"trust": (0.5, 0.8)})]  # no anxiety in TS
    result = blend_with_trajectory(detected, ts)
    trust_trait = [t for t in result.traits if t.name == "trust"][0]
    anxiety_trait = [t for t in result.traits if t.name == "anxiety"][0]
    assert trust_trait.value != 0.7  # blended
    assert anxiety_trait.value == 0.3  # unchanged — no TS data


def test_blend_low_confidence_ts():
    detected = _make_detected({"trust": 0.7})
    ts = [_make_ts_result({"trust": (0.3, 0.1)})]  # very low confidence
    result = blend_with_trajectory(detected, ts)
    # ts_weight = 0.1 * 0.4 = 0.04 — almost no influence
    # blended ≈ 0.96 * 0.7 + 0.04 * 0.3 = 0.672 + 0.012 = 0.684
    assert abs(result.traits[0].value - 0.684) < 0.01


def test_blend_preserves_profile_metadata():
    detected = _make_detected({"trust": 0.7})
    detected.id = "my_profile"
    detected.version = "2.8"
    ts = [_make_ts_result({"trust": (0.5, 0.8)})]
    result = blend_with_trajectory(detected, ts)
    assert result.id == "my_profile"
    assert result.version == "2.8"


def test_blend_clamps_to_0_1():
    detected = _make_detected({"trust": 0.95})
    ts = [_make_ts_result({"trust": (0.99, 0.9)})]
    result = blend_with_trajectory(detected, ts)
    assert result.traits[0].value <= 1.0
    assert result.traits[0].value >= 0.0
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_ensemble.py -v`

Expected: `ModuleNotFoundError: No module named 'super_brain.ensemble'` because the module does not exist yet.

**Step 3: Write minimal implementation**

Create `/Users/michael/super-brain/super_brain/ensemble.py`:

```python
"""Ensemble blending of Detector + ThinkSlow trajectory (V2.8)."""

from __future__ import annotations

from super_brain.models import PersonalityDNA, Trait, ThinkSlowResult


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    """Compute weighted mean. Falls back to simple mean if weights sum to 0."""
    total_weight = sum(weights)
    if total_weight == 0:
        return sum(values) / len(values) if values else 0.5
    return sum(v * w for v, w in zip(values, weights)) / total_weight


def blend_with_trajectory(
    detected: PersonalityDNA,
    ts_results: list[ThinkSlowResult],
    max_ts_weight: float = 0.4,
) -> PersonalityDNA:
    """Blend Detector results with ThinkSlow trajectory estimates.

    For each trait:
    - Collect all ThinkSlow estimates and their confidence scores
    - Compute confidence-weighted mean of ThinkSlow estimates
    - Blend: final = det_weight * detector_value + ts_weight * ts_avg
    - ts_weight = mean(confidences) * max_ts_weight (capped)

    Traits not estimated by ThinkSlow keep pure Detector values.

    Args:
        detected: Detector's one-shot PersonalityDNA result.
        ts_results: List of ThinkSlowResult from periodic extraction.
        max_ts_weight: Maximum weight for ThinkSlow (default 0.4 = 40%).

    Returns:
        Blended PersonalityDNA with adjusted trait values.
    """
    if not ts_results:
        return detected

    # Collect ThinkSlow trajectory per trait: name -> [(value, conf), ...]
    ts_trajectory: dict[str, list[tuple[float, float]]] = {}
    for ts in ts_results:
        for trait in ts.partial_profile.traits:
            conf = ts.confidence_map.get(trait.name, trait.confidence)
            ts_trajectory.setdefault(trait.name, []).append((trait.value, conf))

    # Blend each detected trait
    blended_traits = []
    for trait in detected.traits:
        trajectory = ts_trajectory.get(trait.name)
        if trajectory:
            values = [v for v, _ in trajectory]
            confs = [c for _, c in trajectory]
            ts_avg = _weighted_mean(values, confs)
            mean_conf = sum(confs) / len(confs)
            ts_weight = min(mean_conf * max_ts_weight, max_ts_weight)
            det_weight = 1.0 - ts_weight
            blended_value = det_weight * trait.value + ts_weight * ts_avg
            blended_value = max(0.0, min(1.0, blended_value))
            blended_traits.append(Trait(
                dimension=trait.dimension,
                name=trait.name,
                value=round(blended_value, 3),
                confidence=trait.confidence,
                evidence=trait.evidence,
            ))
        else:
            blended_traits.append(trait)

    return PersonalityDNA(
        id=detected.id,
        version=detected.version,
        created=detected.created,
        updated=detected.updated,
        sample_summary=detected.sample_summary,
        traits=blended_traits,
        trait_relations=detected.trait_relations,
    )
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_ensemble.py -v`

Expected: All 10 tests pass.

**Step 5: Commit**

```
git add super_brain/ensemble.py tests/test_ensemble.py
git commit -m "V2.8: add ensemble blending module + tests

Confidence-weighted ThinkSlow trajectory averaging blended with
Detector one-shot results. ThinkSlow capped at 40% max influence.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 2: Wire ensemble into `detect_and_compare()` and `run_eval()`

**Files:**
- Modify: `/Users/michael/super-brain/eval_conversation.py` (`detect_and_compare` at line 1180, `run_eval` call at line 1348)

**Prerequisite:** V2.6 must be applied first (adds `soul` parameter to `detect_and_compare`). If V2.6 is already applied, `detect_and_compare` has signature `(detector, conversation, profile, profile_name, soul=None)`. If V2.6 is NOT yet applied, the signature is `(detector, conversation, profile, profile_name)` and both `soul` and `ts_results` must be added together.

**Step 1: Write the failing test**

Append to `/Users/michael/super-brain/tests/test_ensemble.py`:

```python
def test_detect_and_compare_accepts_ts_results_parameter():
    """Verify detect_and_compare() signature accepts ts_results parameter."""
    import inspect
    from eval_conversation import detect_and_compare
    sig = inspect.signature(detect_and_compare)
    assert "ts_results" in sig.parameters
    assert sig.parameters["ts_results"].default is None
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_ensemble.py::test_detect_and_compare_accepts_ts_results_parameter -v`

Expected: `AssertionError` because `ts_results` is not in the signature yet.

**Step 3: Write minimal implementation**

**3a. Modify `detect_and_compare()` signature** in `/Users/michael/super-brain/eval_conversation.py` (line 1180).

If V2.6 is already applied (signature has `soul` param), change from:

```python
def detect_and_compare(
    detector: Detector,
    conversation: list[dict],
    profile: PersonalityDNA,
    profile_name: str,
    soul: "Soul | None" = None,
) -> dict:
```

to:

```python
def detect_and_compare(
    detector: Detector,
    conversation: list[dict],
    profile: PersonalityDNA,
    profile_name: str,
    soul: "Soul | None" = None,
    ts_results: "list | None" = None,
) -> dict:
```

If V2.6 is NOT yet applied (current state: 4-param signature), change from:

```python
def detect_and_compare(
    detector: Detector,
    conversation: list[dict],
    profile: PersonalityDNA,
    profile_name: str,
) -> dict:
```

to:

```python
def detect_and_compare(
    detector: Detector,
    conversation: list[dict],
    profile: PersonalityDNA,
    profile_name: str,
    soul: "Soul | None" = None,
    ts_results: "list | None" = None,
) -> dict:
```

**3b. Add ensemble blending** after the `detected = detector.analyze(...)` call (after line 1196).

Insert immediately after the `detected = detector.analyze(...)` block and before the `# Build maps` comment:

```python
    # V2.8: Ensemble blend with ThinkSlow trajectory
    if ts_results:
        from super_brain.ensemble import blend_with_trajectory
        detected = blend_with_trajectory(detected, ts_results)
```

The full section should read:

```python
    detected = detector.analyze(
        text=full_text,
        speaker_id=f"eval_{profile_name}",
        speaker_label="Person B",
    )

    # V2.8: Ensemble blend with ThinkSlow trajectory
    if ts_results:
        from super_brain.ensemble import blend_with_trajectory
        detected = blend_with_trajectory(detected, ts_results)

    # Build maps
    original_map = {t.name: t.value for t in profile.traits}
    detected_map = {t.name: t.value for t in detected.traits}
```

Note: If V2.6 is applied first, the `detector.analyze()` call will include `soul_context=soul_ctx` -- the V2.8 blending goes after that call regardless.

**3c. Modify `run_eval()` call** at line 1348.

Change from:

```python
            result = detect_and_compare(detector, conv_slice, profile, profile_name)
```

to:

```python
            result = detect_and_compare(detector, conv_slice, profile, profile_name, ts_results=ts_results)
```

Or if V2.6 is already applied and the call already passes `soul=soul`:

```python
            result = detect_and_compare(detector, conv_slice, profile, profile_name, soul=soul, ts_results=ts_results)
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_ensemble.py -v`

Expected: All 11 tests pass (10 from Task 1 + 1 new signature test).

**Step 5: Commit**

```
git add eval_conversation.py tests/test_ensemble.py
git commit -m "V2.8: wire ensemble blending into detect_and_compare() and run_eval()

detect_and_compare() now accepts ts_results parameter and blends
Detector output with ThinkSlow trajectory via blend_with_trajectory().
run_eval() passes ts_results through to enable ensemble scoring.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 3: Run V2.8 eval and record results

**Files:**
- Run: `/Users/michael/super-brain/eval_conversation.py`
- Modify: `/Users/michael/super-brain/EVAL_HISTORY.md`

**Step 1: Run full test suite to confirm nothing is broken**

Run: `.venv/bin/pytest tests/ -v`

Expected: All tests pass (including the 11 new ensemble tests).

**Step 2: Run the V2.8 eval**

Run:

```bash
cd /Users/michael/super-brain && ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY .venv/bin/python eval_conversation.py 3 20
```

Expected: 3 profiles, 20-turn conversations, detection at checkpoints. The ensemble blending should now be active. Capture full output for recording.

**Step 3: Record results in EVAL_HISTORY.md**

Add a new section to `/Users/michael/super-brain/EVAL_HISTORY.md` with the V2.8 results following the existing format. Include:
- Version label: V2.8 ThinkSlow Trajectory Ensemble
- Date: 2026-03-07
- MAE, within-0.25, within-0.40 counts at each checkpoint
- Per-dimension MAE breakdown
- Brief comparison note vs V2.6 results
- Note on ensemble behavior: how much ThinkSlow shifted final scores

**Step 4: Commit and push**

```
git add EVAL_HISTORY.md
git commit -m "V2.8: record eval results — ThinkSlow trajectory ensemble

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
git push https://$GH_TOKEN@github.com/mozatyin/super-brain.git HEAD
```
