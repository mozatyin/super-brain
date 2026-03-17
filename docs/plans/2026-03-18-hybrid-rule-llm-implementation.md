# Hybrid Rule-Based + LLM Detection Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Split 7 measurable personality traits out of LLM detection into deterministic rule-based scoring, reducing variance and token cost.

**Architecture:** Add `compute_direct_scores()` to `behavioral_features.py` using piecewise linear interpolation. Modify `detector.py` to exclude rule-based traits from LLM batches and merge them into the final pipeline with selective post-processing.

**Tech Stack:** Python 3.12, pytest, existing super_brain package

---

### Task 1: Add `_interpolate()` helper and `compute_direct_scores()` with tests

**Files:**
- Modify: `super_brain/behavioral_features.py:301` (after `_ADJUSTMENT_RULES`)
- Modify: `tests/test_behavioral_features.py:399` (append new test class)

**Step 1: Write the failing tests**

Add to the end of `tests/test_behavioral_features.py`:

```python
from super_brain.behavioral_features import compute_direct_scores, RULE_BASED_TRAITS


class TestComputeDirectScores:
    """Tests for rule-based direct trait scoring."""

    def test_returns_all_seven_traits(self):
        bf = BehavioralFeatures(
            turn_count=10, total_words=500, avg_words_per_turn=50, words_std=40,
            self_ref_ratio=0.05, other_ref_ratio=0.03, hedging_ratio=0.01,
            absolutist_ratio=0.005, question_ratio=0.15, exclamation_ratio=0.1,
            pos_emotion_ratio=0.01, neg_emotion_ratio=0.005,
            politeness_ratio=0.01, curiosity_ratio=0.01, decisiveness_ratio=0.01,
        )
        scores = compute_direct_scores(bf)
        assert set(scores.keys()) == RULE_BASED_TRAITS

    def test_verbosity_short_turns(self):
        bf = BehavioralFeatures(
            turn_count=10, total_words=200, avg_words_per_turn=20, words_std=5,
            self_ref_ratio=0.05, other_ref_ratio=0.03, hedging_ratio=0.01,
            absolutist_ratio=0.005, question_ratio=0.1, exclamation_ratio=0.1,
            pos_emotion_ratio=0.01, neg_emotion_ratio=0.005,
            politeness_ratio=0.01, curiosity_ratio=0.01, decisiveness_ratio=0.01,
        )
        scores = compute_direct_scores(bf)
        assert 0.10 <= scores["verbosity"] <= 0.25  # <40 words/turn

    def test_verbosity_medium_turns(self):
        bf = BehavioralFeatures(
            turn_count=10, total_words=600, avg_words_per_turn=60, words_std=10,
            self_ref_ratio=0.05, other_ref_ratio=0.03, hedging_ratio=0.01,
            absolutist_ratio=0.005, question_ratio=0.1, exclamation_ratio=0.1,
            pos_emotion_ratio=0.01, neg_emotion_ratio=0.005,
            politeness_ratio=0.01, curiosity_ratio=0.01, decisiveness_ratio=0.01,
        )
        scores = compute_direct_scores(bf)
        assert 0.35 <= scores["verbosity"] <= 0.55  # 40-80 range

    def test_verbosity_long_turns(self):
        bf = BehavioralFeatures(
            turn_count=10, total_words=2000, avg_words_per_turn=200, words_std=30,
            self_ref_ratio=0.05, other_ref_ratio=0.03, hedging_ratio=0.01,
            absolutist_ratio=0.005, question_ratio=0.1, exclamation_ratio=0.1,
            pos_emotion_ratio=0.01, neg_emotion_ratio=0.005,
            politeness_ratio=0.01, curiosity_ratio=0.01, decisiveness_ratio=0.01,
        )
        scores = compute_direct_scores(bf)
        assert scores["verbosity"] >= 0.65  # 150+ words/turn

    def test_politeness_low(self):
        bf = BehavioralFeatures(
            turn_count=10, total_words=500, avg_words_per_turn=50, words_std=10,
            self_ref_ratio=0.05, other_ref_ratio=0.03, hedging_ratio=0.01,
            absolutist_ratio=0.005, question_ratio=0.1, exclamation_ratio=0.1,
            pos_emotion_ratio=0.01, neg_emotion_ratio=0.005,
            politeness_ratio=0.002, curiosity_ratio=0.01, decisiveness_ratio=0.01,
        )
        scores = compute_direct_scores(bf)
        assert scores["politeness"] < 0.25

    def test_politeness_high(self):
        bf = BehavioralFeatures(
            turn_count=10, total_words=500, avg_words_per_turn=50, words_std=10,
            self_ref_ratio=0.05, other_ref_ratio=0.03, hedging_ratio=0.01,
            absolutist_ratio=0.005, question_ratio=0.1, exclamation_ratio=0.1,
            pos_emotion_ratio=0.01, neg_emotion_ratio=0.005,
            politeness_ratio=0.040, curiosity_ratio=0.01, decisiveness_ratio=0.01,
        )
        scores = compute_direct_scores(bf)
        assert scores["politeness"] >= 0.58

    def test_decisiveness_high_hedging(self):
        bf = BehavioralFeatures(
            turn_count=10, total_words=500, avg_words_per_turn=50, words_std=10,
            self_ref_ratio=0.05, other_ref_ratio=0.03, hedging_ratio=0.035,
            absolutist_ratio=0.002, question_ratio=0.1, exclamation_ratio=0.1,
            pos_emotion_ratio=0.01, neg_emotion_ratio=0.005,
            politeness_ratio=0.01, curiosity_ratio=0.01, decisiveness_ratio=0.002,
        )
        scores = compute_direct_scores(bf)
        assert scores["decisiveness"] < 0.40  # indecisive

    def test_decisiveness_high_absolutist(self):
        bf = BehavioralFeatures(
            turn_count=10, total_words=500, avg_words_per_turn=50, words_std=10,
            self_ref_ratio=0.05, other_ref_ratio=0.03, hedging_ratio=0.003,
            absolutist_ratio=0.020, question_ratio=0.1, exclamation_ratio=0.1,
            pos_emotion_ratio=0.01, neg_emotion_ratio=0.005,
            politeness_ratio=0.01, curiosity_ratio=0.01, decisiveness_ratio=0.015,
        )
        scores = compute_direct_scores(bf)
        assert scores["decisiveness"] > 0.55  # decisive

    def test_curiosity_many_questions(self):
        bf = BehavioralFeatures(
            turn_count=10, total_words=500, avg_words_per_turn=50, words_std=10,
            self_ref_ratio=0.05, other_ref_ratio=0.03, hedging_ratio=0.01,
            absolutist_ratio=0.005, question_ratio=0.35, exclamation_ratio=0.1,
            pos_emotion_ratio=0.01, neg_emotion_ratio=0.005,
            politeness_ratio=0.01, curiosity_ratio=0.03, decisiveness_ratio=0.01,
        )
        scores = compute_direct_scores(bf)
        assert scores["curiosity"] > 0.55

    def test_hot_cold_high_variance(self):
        bf = BehavioralFeatures(
            turn_count=10, total_words=500, avg_words_per_turn=50, words_std=120,
            self_ref_ratio=0.05, other_ref_ratio=0.03, hedging_ratio=0.01,
            absolutist_ratio=0.005, question_ratio=0.1, exclamation_ratio=0.1,
            pos_emotion_ratio=0.01, neg_emotion_ratio=0.005,
            politeness_ratio=0.01, curiosity_ratio=0.01, decisiveness_ratio=0.01,
        )
        scores = compute_direct_scores(bf)
        assert scores["hot_cold_oscillation"] > 0.55

    def test_hot_cold_low_variance(self):
        bf = BehavioralFeatures(
            turn_count=10, total_words=500, avg_words_per_turn=50, words_std=10,
            self_ref_ratio=0.05, other_ref_ratio=0.03, hedging_ratio=0.01,
            absolutist_ratio=0.005, question_ratio=0.1, exclamation_ratio=0.1,
            pos_emotion_ratio=0.01, neg_emotion_ratio=0.005,
            politeness_ratio=0.01, curiosity_ratio=0.01, decisiveness_ratio=0.01,
        )
        scores = compute_direct_scores(bf)
        assert scores["hot_cold_oscillation"] < 0.30

    def test_self_mythologizing_high_self_ref(self):
        bf = BehavioralFeatures(
            turn_count=10, total_words=500, avg_words_per_turn=50, words_std=10,
            self_ref_ratio=0.12, other_ref_ratio=0.01, hedging_ratio=0.01,
            absolutist_ratio=0.005, question_ratio=0.1, exclamation_ratio=0.1,
            pos_emotion_ratio=0.01, neg_emotion_ratio=0.005,
            politeness_ratio=0.01, curiosity_ratio=0.01, decisiveness_ratio=0.01,
        )
        scores = compute_direct_scores(bf)
        assert scores["self_mythologizing"] > 0.50

    def test_optimism_positive_dominant(self):
        bf = BehavioralFeatures(
            turn_count=10, total_words=500, avg_words_per_turn=50, words_std=10,
            self_ref_ratio=0.05, other_ref_ratio=0.03, hedging_ratio=0.01,
            absolutist_ratio=0.005, question_ratio=0.1, exclamation_ratio=0.1,
            pos_emotion_ratio=0.030, neg_emotion_ratio=0.005,
            politeness_ratio=0.01, curiosity_ratio=0.01, decisiveness_ratio=0.01,
        )
        scores = compute_direct_scores(bf)
        assert scores["optimism"] > 0.55  # pos >> neg

    def test_optimism_negative_dominant(self):
        bf = BehavioralFeatures(
            turn_count=10, total_words=500, avg_words_per_turn=50, words_std=10,
            self_ref_ratio=0.05, other_ref_ratio=0.03, hedging_ratio=0.01,
            absolutist_ratio=0.005, question_ratio=0.1, exclamation_ratio=0.1,
            pos_emotion_ratio=0.003, neg_emotion_ratio=0.025,
            politeness_ratio=0.01, curiosity_ratio=0.01, decisiveness_ratio=0.01,
        )
        scores = compute_direct_scores(bf)
        assert scores["optimism"] < 0.35  # neg >> pos

    def test_empty_features_returns_baselines(self):
        bf = BehavioralFeatures(
            turn_count=0, total_words=0, avg_words_per_turn=0, words_std=0,
            self_ref_ratio=0, other_ref_ratio=0, hedging_ratio=0,
            absolutist_ratio=0, question_ratio=0, exclamation_ratio=0,
            pos_emotion_ratio=0, neg_emotion_ratio=0,
            politeness_ratio=0, curiosity_ratio=0, decisiveness_ratio=0,
        )
        scores = compute_direct_scores(bf)
        assert len(scores) == 7
        # All should be valid floats in [0, 1]
        for v in scores.values():
            assert 0.0 <= v <= 1.0

    def test_interpolation_is_continuous(self):
        """Adjacent inputs should produce scores within 0.05 of each other."""
        for wpt in range(10, 300, 5):
            bf1 = BehavioralFeatures(
                turn_count=10, total_words=wpt*10, avg_words_per_turn=wpt, words_std=10,
                self_ref_ratio=0.05, other_ref_ratio=0.03, hedging_ratio=0.01,
                absolutist_ratio=0.005, question_ratio=0.1, exclamation_ratio=0.1,
                pos_emotion_ratio=0.01, neg_emotion_ratio=0.005,
                politeness_ratio=0.01, curiosity_ratio=0.01, decisiveness_ratio=0.01,
            )
            bf2 = BehavioralFeatures(
                turn_count=10, total_words=(wpt+5)*10, avg_words_per_turn=wpt+5, words_std=10,
                self_ref_ratio=0.05, other_ref_ratio=0.03, hedging_ratio=0.01,
                absolutist_ratio=0.005, question_ratio=0.1, exclamation_ratio=0.1,
                pos_emotion_ratio=0.01, neg_emotion_ratio=0.005,
                politeness_ratio=0.01, curiosity_ratio=0.01, decisiveness_ratio=0.01,
            )
            s1 = compute_direct_scores(bf1)["verbosity"]
            s2 = compute_direct_scores(bf2)["verbosity"]
            assert abs(s1 - s2) < 0.05, f"Discontinuity at wpt={wpt}: {s1} vs {s2}"

    def test_rule_based_traits_constant(self):
        assert RULE_BASED_TRAITS == {
            "verbosity", "politeness", "decisiveness", "curiosity",
            "hot_cold_oscillation", "self_mythologizing", "optimism",
        }
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_behavioral_features.py::TestComputeDirectScores -v`
Expected: FAIL with `ImportError: cannot import name 'compute_direct_scores'`

**Step 3: Implement `_interpolate()`, `RULE_BASED_TRAITS`, and `compute_direct_scores()`**

Add after line 300 in `super_brain/behavioral_features.py` (after `_ADJUSTMENT_RULES` list, before `compute_adjustments`):

```python
# --- Rule-based direct scoring ---

RULE_BASED_TRAITS: set[str] = {
    "verbosity", "politeness", "decisiveness", "curiosity",
    "hot_cold_oscillation", "self_mythologizing", "optimism",
}

# Mapping tables: (breakpoints, scores) for piecewise linear interpolation
_DIRECT_SCORE_MAPS: dict[str, tuple[list[float], list[float]]] = {
    "verbosity": ([0, 40, 80, 150, 300], [0.10, 0.20, 0.42, 0.58, 0.80]),
    "politeness": ([0, 0.005, 0.015, 0.030, 0.060], [0.12, 0.25, 0.38, 0.58, 0.75]),
    "decisiveness": ([-0.03, -0.01, 0, 0.01, 0.03], [0.25, 0.38, 0.50, 0.62, 0.75]),
    "curiosity": ([0, 0.05, 0.15, 0.25, 0.40], [0.28, 0.38, 0.50, 0.62, 0.72]),
    "hot_cold_oscillation": ([0, 30, 60, 100, 160], [0.20, 0.28, 0.38, 0.55, 0.70]),
    "self_mythologizing": ([0, 0.03, 0.06, 0.09, 0.14], [0.25, 0.32, 0.40, 0.52, 0.65]),
    "optimism": ([0, 0.3, 0.5, 0.7, 1.0], [0.22, 0.35, 0.47, 0.60, 0.72]),
}


def _interpolate(value: float, breakpoints: list[float], scores: list[float]) -> float:
    """Piecewise linear interpolation for continuous trait scoring."""
    if value <= breakpoints[0]:
        return scores[0]
    if value >= breakpoints[-1]:
        return scores[-1]
    for i in range(len(breakpoints) - 1):
        if value <= breakpoints[i + 1]:
            t = (value - breakpoints[i]) / (breakpoints[i + 1] - breakpoints[i])
            return scores[i] + t * (scores[i + 1] - scores[i])
    return scores[-1]  # fallback


def compute_direct_scores(features: BehavioralFeatures) -> dict[str, float]:
    """Compute deterministic scores for rule-based traits.

    These traits are fully determined by objective text statistics and do NOT
    need LLM interpretation. Returns {trait_name: score} for all 7 rule-based
    traits, with scores in [0.0, 1.0].
    """
    # Compute composite signals
    decisiveness_signal = (
        features.absolutist_ratio + features.decisiveness_ratio - features.hedging_ratio
    )

    curiosity_signal = features.question_ratio * 0.6 + features.curiosity_ratio * 10.0
    # Scale curiosity_ratio (typically 0.00-0.04) to be comparable with question_ratio (0-0.5)
    curiosity_signal = min(curiosity_signal, 0.40)  # cap at max breakpoint

    pos = features.pos_emotion_ratio
    neg = features.neg_emotion_ratio
    optimism_signal = pos / (pos + neg) if (pos + neg) > 0.001 else 0.5

    return {
        "verbosity": round(_interpolate(
            features.avg_words_per_turn, *_DIRECT_SCORE_MAPS["verbosity"]
        ), 3),
        "politeness": round(_interpolate(
            features.politeness_ratio, *_DIRECT_SCORE_MAPS["politeness"]
        ), 3),
        "decisiveness": round(_interpolate(
            decisiveness_signal, *_DIRECT_SCORE_MAPS["decisiveness"]
        ), 3),
        "curiosity": round(_interpolate(
            curiosity_signal, *_DIRECT_SCORE_MAPS["curiosity"]
        ), 3),
        "hot_cold_oscillation": round(_interpolate(
            features.words_std, *_DIRECT_SCORE_MAPS["hot_cold_oscillation"]
        ), 3),
        "self_mythologizing": round(_interpolate(
            features.self_ref_ratio, *_DIRECT_SCORE_MAPS["self_mythologizing"]
        ), 3),
        "optimism": round(_interpolate(
            optimism_signal, *_DIRECT_SCORE_MAPS["optimism"]
        ), 3),
    }
```

**Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_behavioral_features.py -v`
Expected: ALL PASS (existing + new)

**Step 5: Commit**

```bash
git add super_brain/behavioral_features.py tests/test_behavioral_features.py
git commit -m "feat: add compute_direct_scores() for 7 rule-based traits"
```

---

### Task 2: Modify detector to exclude rule-based traits from LLM batches

**Files:**
- Modify: `super_brain/detector.py:10` (add import)
- Modify: `super_brain/detector.py:341-472` (analyze method)

**Step 1: Write the failing test**

Add to `tests/test_detector.py` (find the existing mock-based test pattern):

```python
def test_rule_based_traits_excluded_from_llm_prompt(monkeypatch):
    """Rule-based traits should not appear in the LLM prompt."""
    from super_brain.behavioral_features import RULE_BASED_TRAITS, BehavioralFeatures

    captured_prompts = []

    class MockResponse:
        content = [type("Block", (), {"text": json.dumps({"reasoning": [], "scores": []})})()]

    class MockMessages:
        def create(self, **kwargs):
            captured_prompts.append(kwargs["messages"][0]["content"])
            return MockResponse()

    class MockClient:
        messages = MockMessages()

    detector = Detector.__new__(Detector)
    detector._client = MockClient()
    detector._model = "test"
    detector._temperature = 0.0

    bf = BehavioralFeatures(
        turn_count=10, total_words=500, avg_words_per_turn=50, words_std=40,
        self_ref_ratio=0.05, other_ref_ratio=0.03, hedging_ratio=0.01,
        absolutist_ratio=0.005, question_ratio=0.15, exclamation_ratio=0.1,
        pos_emotion_ratio=0.01, neg_emotion_ratio=0.005,
        politeness_ratio=0.01, curiosity_ratio=0.01, decisiveness_ratio=0.01,
    )

    result = detector.analyze(
        text="Speaker: Hello world",
        speaker_id="test",
        behavioral_features=bf,
    )

    # Rule-based trait names should NOT appear as trait definitions in prompts
    all_prompt_text = " ".join(captured_prompts)
    for trait_name in RULE_BASED_TRAITS:
        # The trait name might appear in context text but should not appear
        # as a "**trait_name**" definition line in the prompt
        assert f"- **{trait_name}**" not in all_prompt_text, \
            f"Rule-based trait {trait_name} should not be sent to LLM"

    # But rule-based traits SHOULD appear in the final result
    result_names = {t.name for t in result.traits}
    for trait_name in RULE_BASED_TRAITS:
        assert trait_name in result_names, \
            f"Rule-based trait {trait_name} missing from final result"
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_detector.py::test_rule_based_traits_excluded_from_llm_prompt -v`
Expected: FAIL (rule-based traits still in LLM prompt)

**Step 3: Modify `detector.py` analyze()**

Add import at line 10:
```python
from super_brain.behavioral_features import BehavioralFeatures, compute_direct_scores, RULE_BASED_TRAITS
```
(Replace existing `from super_brain.behavioral_features import BehavioralFeatures`)

Modify `analyze()` method. After line 362 (`bf_section = ...`), add:

```python
        # Compute rule-based traits (deterministic, skip LLM)
        direct_scores: dict[str, float] = {}
        if behavioral_features:
            direct_scores = compute_direct_scores(behavioral_features)
```

Modify the batch loop (line 366). After `batch_traits = _get_traits_for_batch(batch_dims)`, add filtering:

```python
            # Exclude rule-based traits from LLM batch
            if direct_scores:
                batch_traits = [t for t in batch_traits if t["name"] not in RULE_BASED_TRAITS]
                if not batch_traits:
                    continue
```

After the LLM batch loop (after line 449, before post-processing), add rule-based trait injection:

```python
        # Inject rule-based traits with high confidence
        if direct_scores:
            from super_brain.catalog import TRAIT_MAP
            for trait_name, score in direct_scores.items():
                trait_info = TRAIT_MAP.get(trait_name, {})
                all_traits.append(
                    Trait(
                        dimension=trait_info.get("dimension", "UNK"),
                        name=trait_name,
                        value=_clamp(score),
                        confidence=0.95,
                        evidence=[Evidence(text="rule-based: computed from text statistics", source="behavioral_features")],
                    )
                )
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_detector.py::test_rule_based_traits_excluded_from_llm_prompt -v`
Expected: PASS

**Step 5: Commit**

```bash
git add super_brain/detector.py tests/test_detector.py
git commit -m "feat: exclude rule-based traits from LLM batches, inject direct scores"
```

---

### Task 3: Make post-processing skip rule-based traits for calibration and adjustments

**Files:**
- Modify: `super_brain/detector.py:451-457` (post-processing section)
- Modify: `super_brain/detector.py` (`_calibrate_known_biases` function)

**Step 1: Write the failing test**

Add to `tests/test_detector.py`:

```python
def test_rule_based_traits_skip_calibration():
    """Rule-based traits should not have calibration corrections applied."""
    from super_brain.detector import _calibrate_known_biases
    from super_brain.behavioral_features import RULE_BASED_TRAITS

    traits = [
        Trait(dimension="EXT", name="verbosity", value=0.42, confidence=0.95,
              evidence=[Evidence(text="rule-based", source="behavioral_features")]),
        Trait(dimension="EXT", name="warmth", value=0.60, confidence=0.70,
              evidence=[Evidence(text="llm", source="input_text")]),
    ]
    calibrated = _calibrate_known_biases(traits)
    cal_map = {t.name: t.value for t in calibrated}

    # verbosity is rule-based → should be unchanged
    assert cal_map["verbosity"] == 0.42
    # warmth is LLM → should be calibrated (has correction scale=0.40, offset=0.20)
    assert cal_map["warmth"] != 0.60
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_detector.py::test_rule_based_traits_skip_calibration -v`
Expected: FAIL (verbosity gets calibrated)

**Step 3: Modify `_calibrate_known_biases` to skip rule-based traits**

In `_calibrate_known_biases` function (around line 590-606), add at the top of the loop:

```python
    for trait in traits:
        # Skip rule-based traits — they don't have LLM bias
        if trait.name in RULE_BASED_TRAITS and trait.evidence and trait.evidence[0].source == "behavioral_features":
            calibrated.append(trait)
            continue
        # ... existing calibration logic
```

Also modify the `apply_adjustments` call site in `analyze()` (around line 451-457). Change to skip rule-based traits:

Replace:
```python
        all_traits = _validate_consistency(all_traits)
        all_traits = _calibrate_known_biases(all_traits)
        all_traits = _bayesian_shrinkage(all_traits)
```

With no change needed — the skip logic is inside `_calibrate_known_biases` itself, and Bayesian shrinkage already won't fire for confidence=0.95 (threshold is 0.60).

For adjustments: modify the caller in `behavioral_features.py:apply_adjustments()` — actually this is called externally in scripts, not inside detector. The design says: "rule-based traits skip post-hoc adjustments." Add to `apply_adjustments`:

Actually, the cleaner approach: in `detector.py analyze()`, when applying adjustments from `compute_adjustments`, filter out rule-based traits from the adjustment dict before applying. But `apply_adjustments` is called externally in scripts (compare_pipeline.py, full_comparison.py, verify_str_fix.py), not inside detector.analyze(). So the filtering must happen at the call sites.

**Simpler approach**: Remove rule-based traits from `_ADJUSTMENT_RULES` and `_CALIBRATION_CORRECTIONS`:

In `_ADJUSTMENT_RULES`, remove rules that target rule-based traits:
- Remove: `("avg_words_per_turn", 60, "below", "verbosity", -0.10)` (line 299)
- Remove: `("hedging_ratio", 0.020, "above", "decisiveness", -0.08)` (line 294)
- Remove: `("absolutist_ratio", 0.010, "above", "decisiveness", 0.06)` (line 295)
- Remove: `("question_ratio", 0.25, "above", "curiosity", 0.06)` (line 297)
- Remove: `("words_std", 80, "above", "hot_cold_oscillation", 0.06)` (line 291)
- Remove: `("words_std", 30, "below", "hot_cold_oscillation", -0.05)` (line 292)

In `_CALIBRATION_CORRECTIONS` dict in detector.py, remove entries for:
- `"verbosity"`, `"politeness"`, `"hot_cold_oscillation"`, `"self_mythologizing"`

(decisiveness, curiosity, and optimism may not have calibration entries — check first)

**Step 4: Run all tests to verify they pass**

Run: `.venv/bin/pytest tests/ -v`
Expected: ALL PASS (256 existing + ~18 new)

**Step 5: Commit**

```bash
git add super_brain/detector.py super_brain/behavioral_features.py tests/test_detector.py
git commit -m "fix: skip calibration/adjustments for rule-based traits"
```

---

### Task 4: Update existing tests that may reference removed adjustment rules

**Files:**
- Modify: `tests/test_behavioral_features.py` (update tests for removed rules)

**Step 1: Run all existing tests to find failures**

Run: `.venv/bin/pytest tests/test_behavioral_features.py -v`

Check if any tests fail because adjustment rules for verbosity, decisiveness, curiosity, or hot_cold_oscillation were removed.

Known tests that will likely need updating:
- `test_verbosity_short_turns` (line 390-399) — expects verbosity adjustment from `avg_words_per_turn < 60`
- `test_decisiveness_hedging_reduces` (line 345-354) — expects decisiveness adjustment from hedging
- `test_decisiveness_absolutist_increases` (line 356-365) — expects decisiveness adjustment from absolutist
- `test_curiosity_high_questions` (line 367-376) — expects curiosity adjustment from questions
- `test_hot_cold_oscillation_high_std` (line 323-332) — expects hot_cold_oscillation adjustment
- `test_hot_cold_oscillation_low_std` (line 334-343) — expects hot_cold_oscillation adjustment

**Step 2: Update tests to reflect that these traits are now rule-based**

Replace the removed-rule tests with assertions that `compute_adjustments` no longer returns adjustments for these traits:

```python
    def test_verbosity_no_longer_adjusted(self):
        """Verbosity is now rule-based — no adjustment rules."""
        features = BehavioralFeatures(
            turn_count=10, total_words=500, avg_words_per_turn=50, words_std=10,
            self_ref_ratio=0.05, other_ref_ratio=0.03, hedging_ratio=0.01,
            absolutist_ratio=0.005, question_ratio=0.1, exclamation_ratio=0.1,
            pos_emotion_ratio=0.01, neg_emotion_ratio=0.005,
            politeness_ratio=0.01, curiosity_ratio=0.01, decisiveness_ratio=0.01,
        )
        adj = compute_adjustments(features)
        assert "verbosity" not in adj

    def test_decisiveness_no_longer_adjusted(self):
        """Decisiveness is now rule-based — no adjustment rules."""
        features = BehavioralFeatures(
            turn_count=10, total_words=500, avg_words_per_turn=50, words_std=10,
            self_ref_ratio=0.05, other_ref_ratio=0.03, hedging_ratio=0.03,
            absolutist_ratio=0.015, question_ratio=0.1, exclamation_ratio=0.1,
            pos_emotion_ratio=0.01, neg_emotion_ratio=0.005,
            politeness_ratio=0.01, curiosity_ratio=0.01, decisiveness_ratio=0.01,
        )
        adj = compute_adjustments(features)
        assert "decisiveness" not in adj

    def test_curiosity_no_longer_adjusted(self):
        """Curiosity is now rule-based — no adjustment rules."""
        features = BehavioralFeatures(
            turn_count=10, total_words=500, avg_words_per_turn=50, words_std=10,
            self_ref_ratio=0.05, other_ref_ratio=0.03, hedging_ratio=0.01,
            absolutist_ratio=0.005, question_ratio=0.30, exclamation_ratio=0.1,
            pos_emotion_ratio=0.01, neg_emotion_ratio=0.005,
            politeness_ratio=0.01, curiosity_ratio=0.01, decisiveness_ratio=0.01,
        )
        adj = compute_adjustments(features)
        assert "curiosity" not in adj

    def test_hot_cold_no_longer_adjusted(self):
        """Hot/cold oscillation is now rule-based — no adjustment rules."""
        features = BehavioralFeatures(
            turn_count=10, total_words=500, avg_words_per_turn=50, words_std=100,
            self_ref_ratio=0.05, other_ref_ratio=0.03, hedging_ratio=0.01,
            absolutist_ratio=0.005, question_ratio=0.1, exclamation_ratio=0.1,
            pos_emotion_ratio=0.01, neg_emotion_ratio=0.005,
            politeness_ratio=0.01, curiosity_ratio=0.01, decisiveness_ratio=0.01,
        )
        adj = compute_adjustments(features)
        assert "hot_cold_oscillation" not in adj
```

**Step 3: Run all tests**

Run: `.venv/bin/pytest tests/ -v`
Expected: ALL PASS

**Step 4: Commit**

```bash
git add tests/test_behavioral_features.py
git commit -m "test: update tests for rule-based trait migration"
```

---

### Task 5: Remove rule-based traits from system prompt calibration instructions

**Files:**
- Modify: `super_brain/detector.py` (system prompt `_SYSTEM_PROMPT`)

**Step 1: Identify lines to remove**

In `_SYSTEM_PROMPT`, the TRAIT-SPECIFIC CALIBRATION section (around lines 100-121) has instructions for traits that are now rule-based. Remove instructions for:
- `verbosity` (DIRECTLY MEASURABLE... — now literally computed by code)
- `politeness` (DIRECTLY COUNTABLE... — now computed by code)
- `hot_cold_oscillation` (Score 0.30-0.40 baseline... — now computed by code)
- `self_mythologizing` (Score 0.30-0.40 baseline... — now computed by code)
- `optimism` (Ratio of positive-to-negative... — now computed by code)

Keep instructions for traits that are still LLM-scored:
- `humor_self_enhancing` (STRICT definition...)
- `fantasy` (Active imagination...)
- `social_dominance` (Score 0.35-0.45...)
- `charm_influence` (Score 0.35-0.45...)
- `trust` (Score 0.40-0.50...)
- `depression` (Look for low energy...)

Also remove `verbosity` and `politeness` from OVER-DETECTED TRAITS ceiling list if present.

**Step 2: Run tests**

Run: `.venv/bin/pytest tests/ -v`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add super_brain/detector.py
git commit -m "refactor: remove rule-based traits from system prompt calibration"
```

---

### Task 6: Full A/B comparison (requires API credits)

**Files:**
- Run: `scripts/full_comparison.py`
- Reference: `data/comparison_old_vs_new.json` (previous results)

**Step 1: Run full comparison**

```bash
source .env && export ANTHROPIC_API_KEY
.venv/bin/python scripts/full_comparison.py
```

**Step 2: Evaluate results against pass criteria**

Check the output:
- Overall cross-pipeline MAE ≤ 0.085 (current baseline: 0.081)
- All 13 dimension MAEs < 0.16
- No new ⚠ dimensions

**Step 3: If PASS → commit everything and push**

```bash
git add -A
git commit -m "results: hybrid rule-based + LLM comparison data"
git push
```

**Step 4: If FAIL → adjust breakpoints in `_DIRECT_SCORE_MAPS` and re-run**

Compare per-trait errors to identify which rule-based trait's mapping is off. Adjust the breakpoints/scores table for that trait, re-run tests, re-run comparison. Repeat until pass criteria met.

Do NOT commit if MAE regresses beyond threshold.
