# Literary Character Progressive Personality Detection

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Validate Super Brain's personality detection against famous literary characters, measuring both final accuracy and progressive convergence speed using Bayesian belief updating.

**Architecture:** ProgressiveDetector wraps existing Detector with per-trait Bayesian priors. Literary dialogue is segmented into fixed-size chunks (10-15 quotes). Each segment updates the prior, producing a convergence curve. Two ground truth sources (crowd-sourced + LLM omniscient) enable cross-validation.

**Tech Stack:** Python 3.12, existing Super Brain Detector, anthropic SDK via OpenRouter, Project Gutenberg texts.

---

## 1. Core Concept

Instead of detecting personality from a single text dump, progressively build a personality profile as more dialogue is observed — mirroring how humans form impressions of others over time.

Inspired by SoulGraph's Dual Soul architecture:
- Each segment detection = Surface observation
- Bayesian merge with accumulated prior = Deep consolidation
- Confidence grows asymptotically with repeated observation

## 2. Components

### 2.1 ProgressiveDetector (`super_brain/progressive.py`)

New module wrapping `Detector` with Bayesian state management.

```python
class ProgressiveDetector:
    """Incrementally builds a personality profile from sequential text segments."""

    def __init__(self, api_key, model="claude-sonnet-4-20250514"):
        self._detector = Detector(api_key, model)
        self._priors: dict[str, tuple[float, float]] = {}  # trait -> (value, confidence)
        self._history: list[dict] = []  # snapshots after each update

    def update(self, text: str, speaker_label: str) -> dict:
        """Detect traits from text segment, merge with priors, return snapshot."""

    def get_profile(self) -> PersonalityDNA:
        """Return current accumulated profile."""

    def get_history(self) -> list[dict]:
        """Return all historical snapshots."""

    def reset(self):
        """Clear all priors and history."""
```

### 2.2 Bayesian Update Formula

```python
def bayesian_update(prior_val, prior_conf, obs_val, obs_conf):
    total_conf = prior_conf + obs_conf
    if total_conf == 0:
        return 0.5, 0.0

    new_val = (prior_val * prior_conf + obs_val * obs_conf) / total_conf
    new_conf = min(0.95, prior_conf + obs_conf * (1.0 - prior_conf) * 0.5)
    return new_val, new_conf
```

Key properties:
- First detection (prior_conf=0): fully adopts observation
- Subsequent: weighted average, high-confidence prior resists change
- Confidence grows asymptotically → 0.95 (always updatable)
- Convergence criterion: >90% traits with confidence >0.70 and last 3 updates change <0.03

### 2.3 Evaluation Script (`eval_literary.py`)

Orchestrates the full experiment:

1. Extract character dialogue from source texts
2. Segment into fixed-size chunks (10-15 quotes per segment)
3. Feed segments sequentially to ProgressiveDetector
4. After each segment: snapshot MAE vs both ground truths
5. Output convergence curves, trait trajectories, final comparison

## 3. Characters

| Character | Source | Expected Segments | Personality Type |
|-----------|--------|-------------------|------------------|
| Scarlett O'Hara | Gone with the Wind | ~17-25 | Emotional + Dark traits extreme |
| Sherlock Holmes | A Study in Scarlet + Adventures | ~15-20 | Cognitive extreme + Low agreeableness |
| Elizabeth Bennet | Pride and Prejudice | ~15-20 | Balanced but distinctly independent |

Selection rationale: Three orthogonal personality profiles covering different trait extremes. All have OpenPsychometrics crowd data available.

## 4. Ground Truth (Dual Source)

### 4.1 OpenPsychometrics Crowd Data
- 3M+ raters, 525 adjective-pair dimensions
- Mapped to our 69 traits via LLM semantic mapping
- Represents "human consensus" on character personality

### 4.2 LLM Omniscient Scoring
- Claude reads full novel text (not just dialogue)
- Scores all 69 traits with access to narration, actions, inner monologue, other characters' perceptions
- Represents "ideal detection with complete information"

Both GTs stored separately, never mixed. Their mutual MAE measures GT reliability itself.

## 5. Segmentation Strategy

Fixed-size segments of 10-15 quotes per segment (not chapter-based).

Rationale:
- Convergence curve X-axis = cumulative dialogue quantity (not chapter number)
- Avoids noise from 0-quote chapters
- Uniform information density per detection call
- More scientifically interpretable convergence curves

## 6. Output Format

### 6.1 Per-Character Outputs

```
data/literary/{character}/
  dialogue.txt          # All extracted quotes
  segments.json         # Segmented quote lists
  gt_crowd.json         # OpenPsychometrics → 69 traits
  gt_llm.json           # LLM omniscient → 69 traits
  convergence.json      # MAE per segment (the main result)
  trajectories.json     # Per-trait value/confidence history
```

### 6.2 Metrics

| Metric | Definition | Purpose |
|--------|-----------|---------|
| MAE convergence curve | MAE vs GT after each segment | Primary result |
| Convergence speed | Segment where MAE first < 0.15 | How fast is "good enough" |
| Trait convergence rank | Order traits reach stability | What's easy/hard to detect from dialogue |
| Final accuracy | MAE after all dialogue | System ceiling |
| GT consistency | MAE between crowd and LLM GTs | Ground truth reliability |

### 6.3 Success Criteria

- Final MAE < 0.20 (better than previous GWTW 0.215)
- At 50% dialogue: MAE < 0.18 (progressive method adds value)
- 40+ traits converged before all dialogue consumed
- Clear convergence curve (monotonically decreasing trend)

## 7. Dialogue Extraction

Use LLM to extract explicitly attributed dialogue from source texts:
- Pattern: `"quote" said/replied/exclaimed CHARACTER`
- Same method used for GWTW Scarlett (258 quotes extracted successfully)
- Sherlock Holmes and Elizabeth Bennet: extract from Project Gutenberg public domain texts

## 8. File Structure

```
super_brain/
  progressive.py          # NEW: ProgressiveDetector module
  detector.py             # EXISTING: used by ProgressiveDetector

eval_literary.py           # NEW: experiment orchestration script

data/literary/
  scarlett/               # dialogue, segments, GTs, results
  sherlock/               # dialogue, segments, GTs, results
  elizabeth/              # dialogue, segments, GTs, results

tests/
  test_progressive.py     # NEW: unit tests for ProgressiveDetector
```
