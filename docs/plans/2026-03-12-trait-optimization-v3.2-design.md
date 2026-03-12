# Super Brain V3.2: Trait Optimization Design

> **Date**: 2026-03-12
> **Baseline**: V3.1 (5-profile MAE 0.196 at 20t, 0.186 at 10t)
> **Target**: MAE < 0.180 (20t), improved trait coverage
> **Approach**: Cut 2 very-hard traits, add 5 easy traits, improve 10 difficult traits via A+B+C

## 1. Problem Analysis

### 1.1 Current State (V3.1, 66 traits)

| Category | Count | MAE Range | Action |
|----------|-------|-----------|--------|
| Easy (strong signal) | 20 | < 0.15 | Keep as-is |
| Moderate | 34 | 0.15-0.25 | Keep as-is |
| Difficult | 10 | 0.25-0.35 | Improve (A+B+C) |
| Very Hard | 2 | > 0.35 | **Remove** |

### 1.2 Traits to Remove (2)

| Trait | Dimension | V3.1 MAE | Reason |
|-------|-----------|----------|--------|
| emotional_expressiveness | EMO | 0.362 | LLM text is always "expressive" — no variance signal in chat |
| emotional_granularity | EMO | 0.354 | Requires observing emotion vocabulary breadth across many contexts; 20 turns insufficient |

Both traits have fundamental observability problems in short conversations. Removing them improves EMO dimension MAE significantly and avoids noise dragging down overall scores.

### 1.3 Difficult Traits to Improve (10)

| Trait | Dimension | Est. MAE | Primary Issue |
|-------|-----------|----------|---------------|
| competence | CON | ~0.288 | Capability is inferred, not observed; needs task-oriented signals |
| mirroring_ability | STR | ~0.282 | LLM naturally mirrors — true signal buried under artifact |
| social_dominance | SOC | ~0.276 | Status signals subtle in casual chat |
| self_consciousness | NEU | ~0.272 | Self-monitoring hard to distinguish from politeness |
| charm_influence | SOC | ~0.268 | Friendliness ≠ charm, but LLM conflates them |
| intuitive_vs_analytical | COG | ~0.262 | "I feel" vs "data shows" too simplistic |
| emotional_regulation | EMO | ~0.258 | Composure is default in text — no dysregulation signal |
| attachment_anxiety | SOC | ~0.252 | Requires relationship context rarely surfaced |
| information_control | STR | ~0.248 | Privacy ≠ strategic control, hard to distinguish |
| hot_cold_oscillation | STR | ~0.245 | Requires multiple engagement shifts across conversation |

### 1.4 New Traits to Add (5)

All 5 have strong, countable text signals and leverage existing behavioral features:

| Trait | Dimension | Detection Method | Expected MAE |
|-------|-----------|-----------------|--------------|
| **verbosity** | EXT | `avg_words_per_turn`, `words_std`, elaboration patterns | < 0.15 |
| **curiosity** | OPN | `question_ratio`, "I wonder"/"how does" frequency, topic exploration breadth | < 0.18 |
| **politeness** | AGR | please/thank you/sorry frequency, softeners, formal markers | < 0.15 |
| **optimism** | EMO | `pos_emotion_ratio` vs `neg_emotion_ratio`, solution-focus, future orientation | < 0.18 |
| **decisiveness** | CON | inverse of `hedging_ratio`, "I will"/"let's do" frequency, definitive statements | < 0.16 |

**Net trait count**: 66 - 2 + 5 = **69 traits**

## 2. Three-Pronged Approach (A + B + C)

### Approach A: Detection Hint Overhaul + Behavioral Rules

**Detection hint improvements** for 10 difficult traits — each gets 3+ concrete behavioral anchors with specific text patterns:

#### competence
- Current: Generic "confident assertions about ability"
- New: "Look for: (1) describing HOW they solved problems step-by-step, (2) mentioning specific skills/tools by name, (3) 'I figured out' / 'I managed to' vs 'someone helped me', (4) volunteering to take on challenges vs waiting to be told"

#### mirroring_ability
- Current: Relies on vocabulary matching (conflated with LLM artifact)
- New: "IGNORE vocabulary matching (LLM artifact). Instead look for: (1) EXPLICITLY commenting on shared experience ('oh I do that too!'), (2) adopting the other person's FRAMING of a topic, (3) adjusting formality level VISIBLY mid-conversation, (4) echoing back the other person's specific words/phrases"

#### social_dominance
- Current: Status references, hierarchy endorsement
- New: "Look for: (1) steering conversation to OWN topics/expertise, (2) one-upping or competitive comparisons, (3) giving unsolicited advice (implicit higher-status), (4) 'well actually' corrections, (5) mentioning achievements/titles without being asked"

#### self_consciousness
- Current: 'sorry for rambling' detection
- New: "Look for: (1) preemptive disclaimers ('this is probably dumb but'), (2) EDITING/QUALIFYING after stating something ('well, I mean, not exactly'), (3) seeking validation ('does that make sense?'), (4) apologizing for taking up space, (5) minimizing own contributions"

#### charm_influence
- Current: "active rapport-building" (too vague)
- New: "Look for: (1) making the OTHER person the focus of positive attention ('you seem like someone who...'), (2) using humor to create warmth (not just being funny), (3) REFRAMING negatives as positives for the other person, (4) strategic compliments that feel earned not generic. Score 0.40-0.50 for normal friendliness"

#### intuitive_vs_analytical
- Current: "I feel like" vs "data shows"
- New: "Look for: (1) HOW they explain decisions — narrative/story vs criteria/framework, (2) whether they cite specific evidence or general impressions, (3) 'it just felt right' vs 'after weighing the options', (4) comfort with ambiguity (intuitive) vs desire for certainty (analytical)"

#### emotional_regulation
- Current: "active emotion management strategies"
- New: "Look for: (1) NAMING emotions explicitly then moving on ('I was frustrated but...'), (2) reframing negative experiences constructively, (3) acknowledging emotional reactions WITHOUT being swept away, (4) distinguishing between 'I feel X' and 'this IS X'. Default 0.45-0.55 for composed conversation"

#### attachment_anxiety
- Current: "reassurance-seeking, fear of rejection"
- New: "Look for: (1) checking if they said something wrong ('hope that didn't come across as...'), (2) over-explaining to avoid misunderstanding, (3) excessive agreement/people-pleasing, (4) mentioning worry about relationships ending or people leaving, (5) reading negative intent into neutral responses"

#### information_control
- Current: "strategic vagueness"
- New: "Look for: (1) answering questions with questions, (2) giving category-level answers when asked for specifics ('I work in tech' vs 'I'm a backend engineer at X'), (3) redirecting personal topics to abstract discussion, (4) noticeably shorter responses on certain topics vs others"

#### hot_cold_oscillation
- Current: "varying engagement levels"
- New: "Look for: (1) message length VARIANCE across turns (some 200 words, some 10 words on similar topics), (2) enthusiasm spikes followed by flat responses, (3) initiating then abandoning topics, (4) warm personal sharing followed by abrupt topic changes to impersonal subjects"

**New behavioral adjustment rules** (15 rules targeting difficult traits):

```python
# competence signals
("avg_words_per_turn", 150, "above", "competence", 0.04)    # elaborate explanations signal competence
("absolutist_ratio", 0.012, "above", "competence", 0.03)     # confident language
("hedging_ratio", 0.025, "above", "competence", -0.05)       # excessive hedging = low competence signal

# social_dominance signals
("question_ratio", 0.30, "above", "social_dominance", -0.05) # asking many questions = deferring
("avg_words_per_turn", 180, "above", "social_dominance", 0.04) # holding the floor
("self_ref_ratio", 0.07, "above", "social_dominance", 0.04)  # self-focused = status-seeking

# self_consciousness signals
("hedging_ratio", 0.025, "above", "self_consciousness", 0.06)  # hedging = self-monitoring
("question_ratio", 0.25, "above", "self_consciousness", 0.04)  # seeking validation

# intuitive_vs_analytical (toward analytical)
("avg_words_per_turn", 160, "above", "intuitive_vs_analytical", 0.05) # elaborate = analytical
("absolutist_ratio", 0.008, "above", "intuitive_vs_analytical", 0.04) # definitive = analytical

# hot_cold_oscillation
("words_std", 80, "above", "hot_cold_oscillation", 0.06)     # high variance = oscillation
("words_std", 30, "below", "hot_cold_oscillation", -0.05)    # low variance = consistent

# decisiveness (new trait)
("hedging_ratio", 0.020, "above", "decisiveness", -0.08)     # hedging = indecisive
("absolutist_ratio", 0.010, "above", "decisiveness", 0.06)   # absolute = decisive

# attachment_anxiety
("question_ratio", 0.30, "above", "attachment_anxiety", 0.04)  # excessive questions = seeking reassurance
```

### Approach B: Minimal Conversation Probes (Ultra-Light)

Add at most **1 natural question** per difficult trait to `trait_topic_map`. These are conversational, not clinical. The conductor's existing logic (ThinkSlow gap detection + force-probe safety valve) handles when to ask — no changes to conductor needed.

| Trait | New Probe Question |
|-------|-------------------|
| competence | "What's something at work you handle better than most people would?" |
| social_dominance | "In group settings, do you usually end up leading or following the flow?" |
| self_consciousness | "Do you ever replay conversations in your head afterward?" |
| intuitive_vs_analytical | "When you have a big decision, do you go with your gut or make a list?" |
| emotional_regulation | "What do you do when something really gets under your skin?" |
| attachment_anxiety | "How do you feel when someone takes a long time to text back?" |
| information_control | "Are you the kind of person who shares everything or plays it close to the chest?" |
| hot_cold_oscillation | (no probe — observable from conversation dynamics) |
| mirroring_ability | (no probe — observable from conversation dynamics) |
| charm_influence | (no probe — observable from conversation dynamics) |

**8 new probes** for the 10 difficult traits (3 are dynamics-observed, no probe needed).

### Approach C: Per-Trait Linear Calibration

Apply `calibrated = clamp(a * raw + b, 0, 1)` using eval data.

**Extend existing `_CALIBRATION_CORRECTIONS`** (currently 6 traits) to cover all 10 difficult traits + 5 new traits. Optimize (a, b) per trait using grid search:
- `a ∈ [0.50, 1.50]`, step 0.05 (21 values)
- `b ∈ [-0.20, 0.20]`, step 0.02 (21 values)
- Search space: 441 candidates per trait
- Objective: minimize MAE on eval data

**Replace current ad-hoc `_CALIBRATION_CORRECTIONS`** format with systematic per-trait calibration for all traits where eval data shows systematic bias.

## 3. Implementation Summary

### Files to Modify

| File | Changes |
|------|---------|
| `catalog.py` | Remove 2 traits, add 5 new trait definitions |
| `detector.py` | Update batch assignments (rebalance 7 batches for 69 traits), update system prompt calibration text, update `_CALIBRATION_CORRECTIONS`, update `_BATCH_CALIBRATION_EXAMPLES` |
| `behavioral_features.py` | Add 15 new adjustment rules, add word lists for new traits (politeness words, decisiveness markers, curiosity phrases) |
| `trait_topic_map.py` | Add 8 new probe questions, add entries for 5 new traits, remove entries for 2 removed traits |
| `think_slow.py` | Update trait count references if hardcoded |
| `eval_detector.py` | Update expected trait count, add new traits to ground truth profiles |

### Batch Rebalancing (69 traits across 7 batches)

| Batch | Dimensions | Traits | Changes |
|-------|-----------|--------|---------|
| 1 | OPN, CON | 13 | +curiosity (OPN), +decisiveness (CON) |
| 2 | EXT, AGR | 13 | +verbosity (EXT), +politeness (AGR) |
| 3 | NEU, HON | 10 | unchanged |
| 4 | DRK, EMO | 9 | -emotional_expressiveness, -emotional_granularity, +optimism |
| 5 | SOC, STR | 10 | unchanged (but updated detection hints) |
| 6 | COG, VAL | 8 | unchanged (but updated detection hints) |
| 7 | HUM | 4 | unchanged |
| | **Total** | **69** | |

### New Trait Definitions

#### verbosity (EXT)
- **Description**: Tendency toward lengthy, elaborate communication vs. brief, concise responses
- **Detection hint**: "Directly measurable: avg_words_per_turn, total elaboration, tendency to add examples/tangents. High = 150+ words/turn with tangents; Low = <50 words/turn, direct answers only"
- **Behavioral**: Primary signal from `avg_words_per_turn` and `words_std`

#### curiosity (OPN)
- **Description**: Active interest in learning, exploring new topics, asking questions
- **Detection hint**: "Look for: question frequency, 'I wonder'/'how does'/'why is' phrases, exploring tangential topics, asking follow-up questions, expressing interest in unfamiliar subjects"
- **Behavioral**: Primary signal from `question_ratio` + new curiosity phrase counter

#### politeness (AGR)
- **Description**: Use of social lubricants, courtesy markers, and considerate language
- **Detection hint**: "Countable: please/thank you/sorry frequency, softeners ('if you don't mind'), formal address, hedging as courtesy vs. uncertainty, apologies"
- **Behavioral**: New `politeness_ratio` feature (thank/please/sorry/excuse per total words)

#### optimism (EMO)
- **Description**: Tendency toward positive framing, seeing opportunities, expecting good outcomes
- **Detection hint**: "Look for: solution-focus vs problem-focus, 'at least'/'on the bright side', future-oriented language, positive reframing of difficulties, 'it'll work out' vs 'it's hopeless'"
- **Behavioral**: Primary signal from `pos_emotion_ratio` - `neg_emotion_ratio` differential

#### decisiveness (CON)
- **Description**: Speed and confidence in making decisions, commitment to choices
- **Detection hint**: "Look for: 'I will'/'I've decided'/'let's do' vs 'maybe'/'I'm not sure'/'it depends', how quickly they commit to positions, whether they revisit/change stances"
- **Behavioral**: Inverse of `hedging_ratio` + `absolutist_ratio`

## 4. Eval Plan

1. Run V3.2 eval with same 5 profiles, 20 turns, n_samples=3
2. Compare against V3.1 baseline (0.196 MAE)
3. Per-trait breakdown focusing on:
   - 10 improved difficult traits: target < 0.22 avg
   - 5 new easy traits: target < 0.17 avg
   - EMO dimension (after removing 2 worst): target < 0.20
4. If overall MAE < 0.180: success
5. If regression on any currently-good trait > 0.03: investigate and fix

## 5. Risk Mitigation

- **New trait calibration cold start**: No historical data for 5 new traits. Mitigation: rely heavily on behavioral features (deterministic) + conservative LLM defaults.
- **Batch rebalancing side effects**: Adding/removing traits from batches may shift LLM attention. Mitigation: keep batch sizes balanced (8-13 traits each).
- **Probe questions feeling quizzy**: Mitigation: only 8 probes across 10 traits, conductor already spaces them out naturally (force-probe only after 6 turns without asking).
- **Over-fitting calibration to 5 profiles**: Mitigation: keep (a,b) ranges conservative, apply LOOCV.
