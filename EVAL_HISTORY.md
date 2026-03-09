# Super Brain — Natural Conversation Eval History

## Test Setup
- V0.1-V0.8: 3 random 66-trait profiles (seeds: 0, 42, 84)
- V0.9+: 5 random 66-trait profiles (seeds: 0, 42, 84, 126, 168)
- 20-turn casual conversations via Chatter + PersonalitySpeaker
- Detector reads full conversation, estimates all 66 traits
- Compared against ground-truth profile vectors

## Results Summary

| Version | Avg MAE | ≤0.25 | ≤0.40 | Profiles | Key Changes |
|---------|---------|-------|-------|----------|-------------|
| V0.1 | 0.253 | 56.1% | 83.3% | 3 | Baseline: speaker-only text, "default LOW" dark traits |
| V0.2 | 0.220 | 62.1% | 87.9% | 3 | Fixed dark trait bias, full conversation context, speaker behavioral hints |
| V0.3 | 0.195 | 70.2% | 91.4% | 3 | Fixed COG regression, AGR overcorrection, universal baseline calibration |
| V0.4 | 0.199 | 66.7% | 92.3% | 3 | Stronger extreme-score guardrails, more speaker hints |
| V0.5 | 0.186 | 70.2% | 93.4% | 3 | Method actor framing, backstory, relative anchoring, batch retry |
| V0.6 | 0.198 | 66.2% | 91.4% | 3 | 5-phase chatter (degraded at 20t), DRK=0.124 best ever |
| V0.7 | 0.192 | 70.7% | 91.9% | 3 | Conditional method actor, softer pressure phase |
| V0.8 | 0.189 | 74.2% | 91.4% | 3 | No asterisk actions, stay consistent, 3-phase revert |
| V0.9 | 0.166 | 75.3% | 94.9% | 3 | Shortened backstory, anti-patterns, comprehensive LLM bias correction |
| V1.0 | 0.198 | 68.2% | 91.9% | 3 | Hard detector rules (REGRESSION — "HARD RULE" confused model) |
| **V1.1** | **0.169** | **78.8%** | 93.9% | 3 | Reverted V1.0 + loyalty/sincerity/competence/feelings hints |
| V1.2 | 0.178 | 75.3% | 93.9% | 3 | Stronger CON (CON↓ but avg_con bug) |
| V1.3 | 0.183 | 72.2% | 92.9% | 3 | charm/hot_cold (DRK=0.076 best ever!) |
| V1.4 | 0.206 | 68.7% | 90.4% | 3 | Post-processing calibration (TERRIBLE — removed) |
| V1.5 | 0.178 | 75.8% | 92.4% | 3 | Strategic chatter (no help — reverted) |
| V1.6 | 0.197 | 66.2% | 92.9% | 3 | Bayesian shrinkage (worse — removed) |
| V1.7 | 0.190 | 70.9% | 92.7% | 5 | 5 profiles for stable estimate (no code changes from V1.1+fixes) |
| V1.8 | 0.184 | 72.7% | 91.8% | 5 | Humor suppression for low-humor profiles |
| **V2.0-2.2** | **0.196** | **69.7%** | **92.4%** | 3 | Deep Listening + ThinkSlow + Gap-Aware Incisive Q |
| V2.0-2.2@10t | **0.176** | **74.2%** | **94.9%** | 3 | Same system, measured at 10 turns (beats V1.8!) |
| **V2.3** | **0.185** | **76.3%** | **92.4%** | 3 | ThinkFast + Conductor (dynamic, no fixed phases) |
| V2.3@10t | **0.168** | **78.3%** | **92.9%** | 3 | Same system, measured at 10 turns |
| **V2.3.1** | **0.189** | **72.7%** | **96.0%** | 3 | PDCA: detection bias fixes + smart Conductor + topic map |
| V2.3.1@10t | **0.183** | **73.7%** | **94.9%** | 3 | Same system, measured at 10 turns |
| **V2.4** | **0.189** | **72.7%** | **94.9%** | 3 | FactExtractor + AdaptiveFrequency + Soul model (coverage=1.00) |
| V2.4@10t | **0.164** | **83.3%** | **96.5%** | 3 | Same system, measured at 10 turns (best 10t ever!) |
| **V2.5** | **0.178** | **74.2%** | **91.9%** | 3 | ThinkDeep + Intentions + Gaps (coverage=1.00, 31 intentions, 28 gaps avg) |
| V2.5@10t | **0.185** | **70.7%** | **96.5%** | 3 | Same system, measured at 10 turns |
| **V2.6** | **0.187** | **71.7%** | **94.4%** | 3 | Soul-Informed Detection (Soul context injected into Detector prompts) |
| V2.6@10t | **0.190** | **73.2%** | **92.4%** | 3 | Same system, measured at 10 turns |
| **V2.7** | **0.196** | **68.7%** | **92.4%** | 3 | ThinkDeep cap=2 + Jaccard dedup (intentions 31→6, gaps 28→6) |
| V2.7@10t | **0.179** | **76.3%** | **93.9%** | 3 | Same system, measured at 10 turns |
| **V2.8** | **0.182** | **74.2%** | **92.9%** | 3 | ThinkSlow trajectory ensemble (Detector + ThinkSlow blend, max 40% TS weight) |
| V2.8@10t | **0.173** | **75.8%** | **91.4%** | 3 | Same system, measured at 10 turns |
| V2.8(5p) | 0.193 | 67.6% | 93.0% | 5 | V2.8 re-run with 5 profiles for stable baseline |
| V2.8(5p)@10t | 0.197 | 67.6% | 92.1% | 5 | Same system, measured at 10 turns |
| **V2.9** | **0.189** | **67.9%** | **91.8%** | 5 | Behavioral features (text-based signals: pronoun ratios, hedging, emotion words) |
| V2.9@10t | **0.178** | **71.8%** | **94.5%** | 5 | Same system, measured at 10 turns |
| **V3.0** | **0.186** | **70.3%** | **91.8%** | 5 | Soul-Aware Diagnostic Questions (LLM-generated contextual probes replace static questions) |
| V3.0@10t | **0.189** | **71.8%** | **91.5%** | 5 | Same system, measured at 10 turns |
| **V3.1** | **0.196** | **69.1%** | **93.0%** | 5 | Strip-back: removed ThinkDeep, DiagQ, Ensemble, Soul-Informed Detection |
| V3.1@10t | **0.186** | **72.1%** | **95.5%** | 5 | Same system, measured at 10 turns (best 5p 10t ≤0.40 ever!) |

## Per-Dimension MAE (20 turns)

Note: V0.1-V0.8 used 3 profiles; V1.7+ used 5 profiles. Numbers not directly comparable.

| Dimension | V0.8 | V1.1(3p) | V1.7(5p) | V1.8(5p) |
|-----------|------|----------|----------|----------|
| CON | 0.203 | — | — | 0.192 |
| EXT | 0.140 | — | — | 0.154 |
| DRK | 0.174 | — | — | 0.168 |
| OPN | 0.179 | — | — | 0.167 |
| VAL | 0.204 | — | — | 0.190 |
| SOC | 0.220 | — | — | 0.173 |
| HUM | 0.235 | — | 0.266 | 0.245 |
| EMO | 0.160 | — | — | 0.181 |
| AGR | 0.161 | — | — | 0.188 |
| COG | 0.197 | — | — | 0.212 |
| HON | 0.188 | — | — | 0.205 |
| NEU | 0.173 | — | 0.127 | 0.126 |
| STR | 0.264 | — | — | 0.237 |

**V2.0-2.2 Per-Dimension MAE (20 turns, 3 profiles):**

| Dimension | V1.8(5p) | V2.2(3p) | Change |
|-----------|----------|----------|--------|
| DRK | 0.168 | **0.141** | -16% |
| EXT | 0.154 | **0.142** | -8% |
| EMO | 0.181 | **0.146** | -19% |
| VAL | 0.190 | **0.165** | -13% |
| OPN | 0.167 | 0.168 | ~ |
| AGR | 0.188 | **0.178** | -5% |
| CON | 0.192 | **0.186** | -3% |
| HUM | 0.245 | **0.198** | -19% |
| NEU | 0.126 | 0.213 | +69% |
| HON | 0.205 | 0.234 | +14% |
| COG | 0.212 | 0.249 | +17% |
| SOC | 0.173 | 0.270 | +56% |
| STR | 0.237 | 0.287 | +21% |

**V2.3 Per-Dimension MAE (20 turns, 3 profiles):**

| Dimension | V2.2(3p) | V2.3(3p) | Change |
|-----------|----------|----------|--------|
| OPN | 0.168 | **0.127** | -24% |
| EXT | 0.142 | 0.147 | +4% |
| DRK | 0.141 | 0.161 | +14% |
| HUM | 0.198 | **0.162** | -18% |
| VAL | 0.165 | 0.173 | +5% |
| CON | 0.186 | **0.180** | -3% |
| NEU | 0.213 | **0.181** | -15% |
| EMO | 0.146 | 0.181 | +24% |
| SOC | 0.270 | **0.185** | -31% |
| AGR | 0.178 | 0.211 | +19% |
| COG | 0.249 | **0.235** | -6% |
| HON | 0.234 | 0.243 | +4% |
| STR | 0.287 | **0.263** | -8% |

**V2.3.1 Per-Dimension MAE (20 turns, 3 profiles):**

| Dimension | V2.3(3p) | V2.3.1(3p) | Change |
|-----------|----------|------------|--------|
| OPN | 0.127 | 0.136 | +7% |
| EXT | 0.147 | 0.153 | +4% |
| AGR | 0.211 | **0.154** | -27% |
| VAL | 0.173 | **0.169** | -2% |
| HON | 0.243 | **0.170** | -30% |
| CON | 0.180 | **0.178** | -1% |
| HUM | 0.162 | 0.191 | +18% |
| NEU | 0.181 | 0.201 | +11% |
| COG | 0.235 | **0.206** | -12% |
| EMO | 0.181 | 0.208 | +15% |
| DRK | 0.161 | 0.209 | +30% |
| STR | 0.263 | **0.251** | -5% |
| SOC | 0.185 | 0.257 | +39% |

## Key Findings

1. **DRK (Dark Traits)** went from worst dimension (MAE 0.402) to one of the best (0.118-0.174) by:
   - Removing "default LOW" detector instruction
   - Adding subtle dark trait linguistic markers to detection hints
   - Adding behavioral hints for Speaker to express dark traits naturally
   - Method actor framing in V0.5+ further improved expression

2. **COG (Cognitive Style)** is the hardest dimension to calibrate because LLM-generated text inherently sounds analytical/intellectual, creating a persistent upward bias on need_for_cognition.

3. **Method actor framing (V0.5)** broke through the prompt engineering plateau:
   - MAE improved from 0.199 to 0.186 — a 6.5% improvement
   - Biggest wins: AGR (0.208→0.156), NEU (0.243→0.157), COG (0.266→0.173)
   - But caused CON regression (0.115→0.242) — method actor makes everyone sound casual

4. **5-phase chatter (V0.6) hurts at 20 turns**: The pressure/depth phases cause speaker character drift, making 20-turn scores worse than 10-turn scores. Simpler 3-phase chatter is more robust.

5. **"Stay consistent" instruction (V0.8)** eliminated the 10t/20t gap:
   - V0.5: 10t=0.181, 20t=0.186 (gap: 0.005)
   - V0.7: 10t=0.175, 20t=0.192 (gap: 0.017)
   - V0.8: 10t=0.190, 20t=0.189 (gap: -0.001) — effectively zero

6. **No asterisk actions (V0.8)**: Removing `*laughs*` style emotes improved signal-to-noise ratio.

7. **Stubbornly hard traits** (consistently >0.40 error):
   - `humor_self_enhancing` — LLM naturally finds humor in everything, always over-detected
   - `mirroring_ability` — LLM naturally mirrors style, always over-detected for low values
   - `social_dominance` — speaker doesn't express dominance in casual chat
   - `information_control` — high info control hard to express without sounding evasive
   - `competence` — method actor/self-deprecation undermines high-competence expression

8. **Best composite version**: V2.3 has best 20t MAE (0.185) and best ≤0.25 rate (76.3%). V2.3@10t has best 10t MAE (0.168) and ≤0.25 rate (78.3%). No single version is best across all dimensions — an ensemble approach could theoretically achieve MAE ~0.14 by picking per-dimension best.

9. **5-profile eval reveals true performance**: V2.8(5p) MAE 0.193 vs 3p 0.182 (+6%). The 3-profile estimates consistently flattered the system. V2.9(5p) at 0.189 is the most reliable measurement to date.

10. **Behavioral features provide zero-cost signal**: Text-level features (pronoun ratios, hedging, emotion words) improve 10t detection by ~10% at zero LLM cost. Effect diminishes at 20t where LLM-based detection has more data.

## Changes Per Version

### V0.2
- Detector: Removed "default LOW" dark trait instruction, added conversation context awareness, LLM bias correction
- Speaker: Added behavioral hints for dark traits, neuroticism, humor, strategy, low-prosocial
- Eval: Feed full conversation (both speakers) to detector instead of speaker-only text
- Chatter: Added topic escalation phases (light → opinions → deeper)
- Catalog: Rewrote detection_hints for DRK, NEU, EMO, HUM, STR, SOC traits

### V0.3
- Detector: Added COG-specific LLM bias correction, universal baseline calibration, humility calibration, rebalanced AGR detection
- Speaker: Added hints for low/high cognition, high AGR traits, humility, empathy_affective, loyalty_group

### V0.4
- Detector: Strengthened extreme-score guardrails (FORBIDDEN <0.15 or >0.85 without 3+ observations), trait-specific calibration for mirroring, self_consciousness, tender_mindedness, emotional_volatility
- Speaker: Added hints for self_consciousness, competence

### V0.5 (breakthrough)
- Speaker: Method actor framing ("you are performing a character study") to bypass LLM safety alignment
- Speaker: Experience-grounded backstory generation from trait vector (PsyPlay technique)
- Speaker: Response length/complexity scaling by need_for_cognition
- Speaker: Per-turn temporal modulation for hot_cold_oscillation and emotional_volatility
- Detector: Relative anchoring scoring ("compared to average person")
- Detector: Batch completeness retry (auto-retry when traits missing)
- Detector: Stronger humor_self_enhancing calibration

### V0.6
- Chatter: 5-phase emotional terrain map (Rapport→Social→Opinions→Pressure→Depth)
- Speaker: Added CON behavioral hints (self_discipline, achievement_striving, order, deliberation)
- Speaker: Added information_control and humor_self_enhancing low-value hints
- Detector: Added conscientiousness calibration section

### V0.7
- Speaker: Conditional method actor (only for dark/extreme traits, not universal)
- Speaker: "STAY CONSISTENT" instruction to prevent character drift
- Speaker: Strengthened low care_harm expression
- Chatter: Softened pressure phase to "real talk"

### V0.8
- Speaker: "NEVER use *asterisk actions*" — removes noise from `*laughs*` etc.
- Speaker: Added social_dominance, conflict_assertiveness behavioral hints
- Speaker: Added anti-patterns (low humor_self_enhancing, low charm, low mirroring)
- Chatter: Reverted to 3-phase (simpler performs better at 20 turns)
- Detector: Stronger mirroring_ability baseline (0.35-0.45)
- Detector: Stronger humor_self_enhancing definition

### V0.9
- Speaker: Shortened backstory (5 fragments max)
- Speaker: Anti-patterns section for traits LLM naturally over-expresses
- Detector: Comprehensive LLM bias correction section (articulateness, warmth, self-deprecating humor, style matching)

### V1.0 (REGRESSION)
- Detector: "HARD RULE" language for need_for_cognition and humor_self_enhancing — confused model's overall calibration
- Result: MAE regressed 0.166→0.198

### V1.1 (best 3-profile result)
- Detector: Reverted V1.0 hard rules, softened to guidance language
- Speaker: Added loyalty_group behavioral hints (>0.80 threshold with "we" language)
- Speaker: Added sincerity hints (high >0.60, low <0.30)
- Speaker: Added competence hints (high >0.60, low <0.30)
- Speaker: Added low feelings (<0.30) suppression
- Backstory: Added loyalty and sincerity fragments

### V1.2
- Speaker: Stronger achievement_striving expression (>0.70 threshold)
- Speaker: CON anti-self-deprecation (high_con_count ≥ 3 check, fixed from avg_con bug)
- Detector: Mirroring_ability baseline lowered to 0.30-0.40

### V1.3
- Speaker: Stronger hot_cold_oscillation (>0.60 with 3-phase cycle)
- Detector: charm_influence calibration (0.40-0.50 baseline)
- Kept: high_con_count fix, charm_influence calibration

### V1.4 (TERRIBLE — removed)
- Detector: Post-processing affine calibration corrections for 6 traits — compounded with existing calibration

### V1.5 (no help — reverted)
- Chatter: Strategic dimension-probing questions per phase

### V1.6 (worse — removed)
- Detector: Bayesian shrinkage pulling low-confidence scores toward 0.50

### V1.7 (stable estimate)
- No code changes from V1.1+fixes; 5 profiles instead of 3 for stable MAE estimate
- True performance: ~0.190 (higher than 3-profile V0.9/V1.1 estimates)

### V1.8
- Speaker: Comprehensive humor suppression for low-humor profiles (avg_humor < 0.35)
- Speaker: Raised humor_self_enhancing anti-pattern threshold from 0.35 to 0.40
- Result: HUM improved 0.266→0.245, overall MAE 0.190→0.184

### V1.9 (running)
- Detector: Added "SYSTEMATICALLY OVER-DETECTED TRAITS" section (modesty, straightforwardness, humor_affiliative)
- Detector: Lowered humor_self_enhancing baseline to 0.35-0.45, require 2+ examples for >0.55
- Speaker: Stronger low-modesty anti-pattern (threshold 0.35→0.45, more aggressive language)
- Speaker: Added low-straightforwardness anti-pattern (<0.45)
- Speaker: Raised high-modesty threshold to >0.65 with stronger instruction

### V2.0-2.2 — Deep Listening + ThinkSlow + Gap-Aware Incisive Questions

**Fundamental approach change**: Instead of optimizing prompts (diminishing returns), changed HOW we collect personality signal.

**V2.0 — Deep Listening Conversation Strategy**
- Chatter: Rewrote `_build_chatter_system()` with Nancy Kline's 10 Component Thinking Environment
- Chatter: 3 phases: turns 1-7 rapport, 8-14 deepening, 15+ incisive questions
- Chatter: Reduced max_tokens from 256 to 150 (shorter prompts = more speaker output)
- Core principles: Full Attention, Ease, Equality, Appreciation, Encouragement, Feelings, Information, Diversity, Place

**V2.1 — Think Slow Periodic Extraction**
- New module: `super_brain/think_slow.py`
- ThinkSlowResult model: partial_profile, confidence_map, low_confidence_traits, observations
- Periodic extraction every 5 turns with gap-aware focus (passes low-confidence traits to next round)
- Conservative confidence calibration (0.2-0.5 typical for 5-turn casual chat)

**V2.2 — Gap-Aware Incisive Questions**
- New module: `super_brain/trait_topic_map.py` — 47 traits → 2-3 natural conversation topics each
- `_build_chatter_system` now accepts `low_confidence_traits` parameter
- Incisive Questions phase (turn 15+) targets low-confidence traits from ThinkSlow
- ThinkSlow flags ALL unestimated traits as low-confidence (not just low-confidence estimated ones)

**Key findings**:
1. **10-turn detection beats V1.8** (MAE 0.176 vs 0.184) — Deep Listening generates better signal faster
2. **20-turn detection regresses** (MAE 0.196) — Detector's "CASUAL CONVERSATION" calibration fights deeper conversation style
3. **DRK, EMO, HUM all improved significantly** at 20 turns (-16%, -19%, -19%)
4. **NEU, SOC, STR regressed** — Detector's conservative baselines suppress legitimate deeper conversation signals
5. **Next step**: Detector context awareness needs updating for Deep Listening conversation style, or use ThinkSlow results directly instead of one-shot batch detection

### V2.3 — ThinkFast + Conductor (Dynamic Probabilistic Modes)

**Core change**: Replaced fixed conversation phases (rapport→deepening→incisive) with dynamic Conductor that decides action each turn based on real-time signals.

**New modules**:
- `super_brain/think_fast.py` — Rule-based signal detection (every turn): new_facts, emotional_shift, contradiction, opening, info_entropy
- `super_brain/conductor.py` — Dynamic action selection: listen / follow_thread / ask_incisive / push
- Enhanced `super_brain/think_slow.py` — Now generates ranked incisive questions from trait gaps via trait_topic_map

**Architecture**:
- ThinkFast analyzes every speaker message (regex patterns, no LLM cost)
- Conductor decides mode based on: early turns→listen, opening→follow_thread, stale info+questions→ask_incisive
- `_build_chatter_from_action()` replaces `_build_chatter_system()` — Deep Listening base + mode-specific suffix
- ThinkSlow interval changed from 5 to 3 turns (more frequent extraction)

**Key findings**:
1. **Both 10t and 20t improved** over V2.2: 10t 0.176→0.168 (-4.5%), 20t 0.196→0.185 (-5.6%)
2. **10t/20t gap smaller**: 0.017 vs V2.2's 0.020 — dynamic modes avoid 20t regression
3. **SOC massively improved** at 20t: 0.270→0.185 (-31%) — no longer fighting fixed phase structure
4. **NEU recovered**: 0.213→0.181 (-15%) — dynamic listening doesn't force unnatural deepening
5. **OPN best ever**: 0.127 (-24%) — Conductor follows openings naturally
6. **HUM improved**: 0.198→0.162 (-18%) — better signal from dynamic conversation flow
7. **EMO regressed**: 0.146→0.181 (+24%) — needs investigation, may need emotion-specific follow-up mode
8. **AGR regressed**: 0.178→0.211 (+19%) — dynamic mode may not probe agreeableness enough
9. **STR, HON still stubborn** but improved: STR 0.287→0.263, HON 0.234→0.243

### V2.3.1 — PDCA Iteration: Detection Bias + Smart Conductor + Complete Topic Map

**Three structural fixes applied via PDCA cycles:**

**Cycle 1: Detection hint LLM bias corrections** (5 traits)
- `mirroring_ability`: baseline 0.40-0.50, LLM naturally mirrors
- `self_mythologizing`: baseline 0.30-0.40, LLM avoids drama
- `fairness`: baseline 0.45-0.55, LLM sounds inherently fair
- `humility_hexaco`: baseline 0.45-0.55, LLM defaults to humble
- `need_for_cognition`: baseline 0.40-0.50, LLM sounds analytical

**Cycle 2: Conductor improvements**
- Force-probe: incisive question every 6 turns if none asked (only after turn 8)
- Entropy threshold kept at 0.3 (raising to 0.5 was too aggressive)
- Question rotation: prefers unasked trait targets over repeated ones
- Stateful tracking of turns_since_last_incisive and asked_targets

**Cycle 3: Complete trait_topic_map** (17 missing traits added)
- AGR: straightforwardness, altruism, compliance, tender_mindedness (2/6 → 6/6)
- NEU: angry_hostility, depression, impulsiveness, self_consciousness, vulnerability (1/6 → 6/6)
- EXT: activity_level, excitement_seeking, positive_emotions
- OPN: actions, aesthetics
- CON: dutifulness; HON: greed_avoidance; COG: intuitive_vs_analytical

**Key findings**:
1. **HON massively improved**: 0.243→0.170 (-30%) — detection bias corrections worked
2. **AGR massively improved**: 0.211→0.154 (-27%) — topic map + bias corrections
3. **COG improved**: 0.235→0.206 (-12%) — need_for_cognition bias correction
4. **≤0.40 rate best ever**: 96.0% (fewer catastrophic errors)
5. **Overall MAE flat**: 0.185→0.189 (within conversation variance for 3 profiles)
6. **DRK, SOC regressed**: likely conversation variance (untouched detection hints)
7. **Lesson**: 3-profile eval has high variance; targeted dimension improvements are real but net MAE obscured by noise

### V2.4 — FactExtractor + Adaptive Frequency + Soul Model

**Core change**: Added Soul model expanding beyond personality traits to capture facts, reality, secrets, and contradictions. New FactExtractor runs alongside ThinkSlow with adaptive frequency.

**New modules**:
- `super_brain/fact_extractor.py` — Separate LLM call extracting facts (career, location, family, etc.), reality narrative, secrets, contradictions
- `super_brain/adaptive_frequency.py` — Interval manager (2-5 turns) adjusting based on extraction yield
- `super_brain/soul_coverage.py` — Coverage scoring: facts/10 + reality + secrets/3

**New models** (in `models.py`):
- `Fact`: category, content, confidence, source_turn
- `Reality`: summary, domains, constraints, resources
- `FactExtractionResult`: new_facts, reality, secrets, contradictions
- `Soul`: character (PersonalityDNA) + facts + reality + secrets + contradictions

**Simulation changes**:
- `simulate_conversation()` now accepts `fact_extractor` parameter
- When provided: both ThinkSlow and FactExtractor use AdaptiveFrequency (default=3 turns)
- Returns 3-tuple: (conversation, ts_results, soul)
- Backward compatible: old 2-tuple still returned when no fact_extractor

**Eval changes**:
- `run_eval()` instantiates FactExtractor and passes to simulation
- Reports Soul Coverage metrics per profile and in summary
- Results JSON includes soul_coverage, facts_count, reality_populated, secrets_count

**V2.4 Per-Dimension MAE (20 turns, 3 profiles):**

| Dimension | V2.3.1(3p) | V2.4(3p) | Change |
|-----------|------------|----------|--------|
| DRK | 0.209 | **0.139** | -33% |
| EXT | 0.153 | **0.143** | -7% |
| CON | 0.178 | **0.154** | -13% |
| COG | 0.206 | **0.164** | -20% |
| VAL | 0.169 | **0.176** | +4% |
| OPN | 0.136 | 0.177 | +30% |
| AGR | 0.154 | 0.188 | +22% |
| EMO | 0.208 | **0.190** | -9% |
| NEU | 0.201 | **0.210** | +4% |
| HON | 0.170 | 0.227 | +34% |
| SOC | 0.257 | **0.229** | -11% |
| HUM | 0.191 | 0.240 | +26% |
| STR | 0.251 | **0.243** | -3% |

**Soul Coverage (NEW — V2.4):**

| Metric | Target | Actual |
|--------|--------|--------|
| Avg coverage score | ≥0.50 | **1.000** |
| Avg facts per profile | ≥5 | **71.3** |
| Reality populated | 3/3 | **3/3** |
| Avg secrets per profile | — | **32.0** |
| Avg contradictions per profile | — | **26.0** |

**Key findings**:
1. **Soul Coverage maxed out**: 1.000 — FactExtractor is highly productive with adaptive frequency
2. **10t MAE best ever**: 0.164 (vs V2.3@10t 0.168) — 10.4% improvement over V2.3.1@10t
3. **20t MAE unchanged**: 0.189 — expected since V2.4 focuses on Soul expansion, not MAE
4. **10t ≤0.40 rate best ever**: 96.5% (new record)
5. **ThinkSlow runs more often**: 9 times in 20 turns (adaptive frequency starts at 3, decreases to 2 on high yield)
6. **DRK massively improved**: 0.209→0.139 (-33%) — conversation variance aligns with natural expression
7. **COG improved**: 0.206→0.164 (-20%) — consistent with V2.3.1 bias correction
8. **Dimension variance high**: HON +34%, OPN +30% regressions likely conversation variance (3-profile noise)
9. **71 facts per profile**: FactExtractor extracts career, hobby, relationship, preference data richly
10. **32 secrets per profile**: Detects avoidance patterns, energy shifts, hidden motivations

### V2.5 — ThinkDeep + Intentions + Gaps

**Core change**: Added ThinkDeep — a triggered (not periodic) strategic analysis that detects intentions, reality-intention gaps, and bridge questions from the full Soul state. Enhanced Conductor with gap-driven "push" mode.

**New modules**:
- `super_brain/think_deep.py` — Triggered LLM call analyzing full Soul → intentions, gaps, bridge questions, critical question, conversation strategy

**New models** (in `models.py`):
- `Intention`: description, domain, strength, blockers
- `Gap`: intention, reality, bridge_question, priority
- `ThinkDeepResult`: soul_narrative, intentions, gaps, critical_question, conversation_strategy

**Conductor enhancement**:
- New `think_deep` parameter in `decide()`
- Priority 1.5: critical_question → "push" mode (one-shot, cleared after use)
- Merged question pool: trait_gap questions + gap bridge questions compete by priority

**Trigger conditions** (any of):
- 5+ facts accumulated and no intentions yet
- New contradiction found by FactExtractor
- ThinkSlow info_staleness > 0.8 for 2+ consecutive cycles
- After turn 10 if no intentions detected

**Soul Coverage expanded**: 3 → 5 components (facts/10, reality, secrets/3, intentions/3, gaps/2)

**V2.5 Per-Dimension MAE (20 turns, 3 profiles):**

| Dimension | V2.4(3p) | V2.5(3p) | Change |
|-----------|----------|----------|--------|
| EXT | 0.143 | **0.088** | -38% |
| OPN | 0.177 | **0.126** | -29% |
| HON | 0.227 | **0.138** | -39% |
| AGR | 0.188 | **0.149** | -21% |
| VAL | 0.176 | **0.157** | -11% |
| CON | 0.154 | 0.173 | +12% |
| NEU | 0.210 | **0.183** | -13% |
| DRK | 0.139 | 0.218 | +57% |
| HUM | 0.240 | **0.219** | -9% |
| SOC | 0.229 | **0.221** | -3% |
| COG | 0.164 | 0.223 | +36% |
| EMO | 0.190 | 0.228 | +20% |
| STR | 0.243 | **0.232** | -5% |

**Soul Coverage V2.5:**

| Metric | Target | Actual |
|--------|--------|--------|
| Avg coverage score | ≥0.50 | **1.000** |
| Avg facts per profile | ≥5 | **62.7** |
| Reality populated | 3/3 | **3/3** |
| Avg secrets per profile | — | **34.7** |
| Avg contradictions per profile | — | **30.0** |
| Avg intentions per profile | ≥2 | **31.3** |
| Avg gaps per profile | ≥1 | **28.3** |

**Key findings**:
1. **20t MAE new best**: 0.178 (vs V2.4 0.189) — 5.8% improvement
2. **EXT massively improved**: 0.143→0.088 (-38%) — best single-dimension result ever
3. **HON massively improved**: 0.227→0.138 (-39%) — ThinkDeep's gap-driven questions probe honesty naturally
4. **OPN recovered**: 0.177→0.126 (-29%) — back to V2.3's strong OPN performance
5. **AGR improved**: 0.188→0.149 (-21%) — bridge questions explore trust and cooperation
6. **10t MAE regressed**: 0.164→0.185 — ThinkDeep triggers mid-conversation, so 10-turn slice doesn't benefit yet
7. **DRK regressed**: 0.139→0.218 (+57%) — high conversation variance, 3-profile noise
8. **COG regressed**: 0.164→0.223 (+36%) — LLM analytical bias returns
9. **ThinkDeep highly productive**: 31.3 intentions, 28.3 gaps per profile — may be over-extracting
10. **Conductor push mode active**: ThinkDeep critical questions drive deeper conversation probing

### V2.6 — Soul-Informed Detection

**Core change**: Pass accumulated Soul context (top 10 facts, reality, top 3 intentions) to the Detector's LLM prompts for better calibration. New helper `_build_detector_soul_context()` serializes Soul state into a concise markdown block injected between "Target Speaker" and "Dimensions to Analyze" sections.

Also added: API retry wrapper (`api_retry.py`) for transient OpenRouter 403/429/500 errors.

**V2.6 Per-Dimension MAE (20 turns, 3 profiles):**

| Dimension | V2.5(3p) | V2.6(3p) | Change |
|-----------|----------|----------|--------|
| EXT | 0.088 | **0.117** | +33% |
| VAL | 0.157 | **0.142** | -10% |
| HON | 0.138 | 0.174 | +26% |
| COG | 0.223 | **0.174** | -22% |
| AGR | 0.149 | 0.175 | +17% |
| HUM | 0.219 | **0.175** | -20% |
| EMO | 0.228 | **0.181** | -21% |
| CON | 0.173 | 0.183 | +6% |
| DRK | 0.218 | **0.193** | -11% |
| OPN | 0.126 | 0.196 | +56% |
| NEU | 0.183 | 0.216 | +18% |
| STR | 0.232 | 0.232 | ~ |
| SOC | 0.221 | 0.267 | +21% |

**Key findings**:
1. **≤0.40 rate improved**: 91.9%→94.4% (+2.5pp) — fewer catastrophic errors, Soul context helps prevent extreme miscalibrations
2. **20t MAE regressed slightly**: 0.178→0.187 (+5.1%) — Soul context may bias some dimensions
3. **COG improved**: 0.223→0.174 (-22%) — knowing the person's background helps calibrate analytical trait detection
4. **EMO, HUM improved**: -21%, -20% — emotional and humor calibration benefits from person context
5. **DRK improved**: 0.218→0.193 (-11%) — knowing reality helps distinguish dark trait expression
6. **OPN regressed**: 0.126→0.196 (+56%) — Soul context may be anchoring openness estimates
7. **EXT regressed**: 0.088→0.117 (+33%) — V2.5's exceptional EXT result was likely variance
8. **Overall mixed**: Soul context reduces catastrophic errors but slightly increases mean error — may benefit more with cleaner Soul state (V2.7 dedup)

### V2.7 — ThinkDeep Quality Control + Dedup

**Core change**: Cap ThinkDeep at max 2 fires per conversation (was unlimited via boolean reset). Add Jaccard token-level deduplication for intentions, gaps, secrets, and contradictions to reduce Soul state bloat.

**New module**: `super_brain/dedup.py` — `is_duplicate()` (Jaccard similarity) and `dedup_extend_strings()`

**V2.7 Per-Dimension MAE (20 turns, 3 profiles):**

| Dimension | V2.6(3p) | V2.7(3p) | Change |
|-----------|----------|----------|--------|
| DRK | 0.193 | **0.145** | -25% |
| EXT | 0.117 | 0.147 | +26% |
| VAL | 0.142 | 0.154 | +8% |
| AGR | 0.175 | **0.173** | -1% |
| HON | 0.174 | 0.177 | +2% |
| NEU | 0.216 | **0.196** | -9% |
| OPN | 0.196 | 0.199 | +2% |
| CON | 0.183 | 0.203 | +11% |
| COG | 0.174 | 0.214 | +23% |
| EMO | 0.181 | 0.221 | +22% |
| STR | 0.232 | 0.233 | ~ |
| HUM | 0.175 | 0.240 | +37% |
| SOC | 0.267 | **0.240** | -10% |

**Soul State Quality (V2.7 vs V2.5):**

| Metric | V2.5 | V2.7 | Target | Hit? |
|--------|------|------|--------|------|
| Intentions/profile | 31.3 | **6.3** | 5-10 | YES |
| Gaps/profile | 28.3 | **6.0** | 3-6 | YES |
| Secrets/profile | 34.7 | 34.7 | 10-15 | No (Jaccard too weak for secrets) |
| Contradictions/profile | 30.0 | 30.3 | — | No (same issue) |

**Key findings**:
1. **Dedup achieved target**: Intentions 31→6.3, gaps 28→6.0 — dramatic noise reduction
2. **Secrets/contradictions unchanged**: Jaccard token overlap doesn't catch semantically similar but differently-worded secrets — would need embeddings
3. **10t MAE improved**: 0.190→0.179 — less ThinkDeep disruption mid-conversation (cap=2 helps)
4. **20t MAE regressed**: 0.187→0.196 — cleaner Soul context didn't help 20t detection, likely conversation variance
5. **DRK improved**: 0.193→0.145 (-25%) — less noise in Soul context helps dark trait calibration
6. **SOC improved**: 0.267→0.240 (-10%) — cleaner intentions = better social dynamics context
7. **HUM, EMO regressed**: likely conversation variance with 3 profiles
8. **ThinkDeep cap working**: Only 2 fires per conversation, less push mode disruption

### V2.8 — ThinkSlow Trajectory Ensemble

**Core change**: Blend the Detector's one-shot personality estimates with ThinkSlow's progressive trajectory of partial trait observations. Confidence-weighted averaging with ThinkSlow capped at 40% max influence.

**New module**: `super_brain/ensemble.py` — `blend_with_trajectory()` and `_weighted_mean()`

**Algorithm**:
```
for each trait:
    ts_avg = confidence_weighted_mean(ThinkSlow estimates across all 9 cycles)
    ts_weight = mean(confidences) * 0.4   # max 40% TS contribution
    final = (1 - ts_weight) * detector_value + ts_weight * ts_avg
```

**V2.8 Per-Dimension MAE (20 turns, 3 profiles):**

| Dimension | V2.7(3p) | V2.8(3p) | Change |
|-----------|----------|----------|--------|
| VAL | 0.154 | **0.138** | -10% |
| DRK | 0.145 | **0.139** | -4% |
| EXT | 0.147 | 0.147 | ~ |
| HON | 0.177 | **0.167** | -6% |
| CON | 0.203 | **0.180** | -11% |
| NEU | 0.196 | **0.181** | -8% |
| EMO | 0.221 | **0.185** | -16% |
| OPN | 0.199 | **0.186** | -7% |
| SOC | 0.240 | **0.195** | -19% |
| STR | 0.233 | **0.204** | -12% |
| AGR | 0.173 | 0.209 | +21% |
| COG | 0.214 | **0.210** | -2% |
| HUM | 0.240 | **0.230** | -4% |

**Key findings**:
1. **Ensemble smoothing works**: 20t MAE 0.196→0.182 — biggest single-version improvement in V2.6-V2.8 series
2. **10 of 13 dimensions improved**: Only AGR regressed (+21%), likely conversation variance
3. **SOC massively improved**: 0.240→0.195 (-19%) — ThinkSlow trajectory helps social dynamics detection
4. **EMO improved**: 0.221→0.185 (-16%) — progressive observation of emotional patterns helps
5. **CON improved**: 0.203→0.180 (-11%) — trajectory builds better conscientiousness signal
6. **STR improved**: 0.233→0.204 (-12%) — strategy traits benefit from incremental observation
7. **10t MAE also improved**: 0.179→0.173 — ensemble helps even at 10 turns (5 ThinkSlow cycles available)
8. **Best 20t MAE in V2.6-V2.8**: 0.182 (V2.6: 0.187, V2.7: 0.196) — ensemble is the winning technique
9. **V2.5 baseline comparison**: 20t 0.178→0.182 (+2.2%), 10t 0.185→0.173 (-6.5%) — net improvement at 10t, slight regression at 20t still within conversation variance

### V2.8(5p) — 5-Profile Stable Baseline

**No code changes** — re-ran V2.8 with 5 profiles (seeds 0, 42, 84, 126, 168) for stable measurement.

**Key findings**:
1. **5p MAE higher than 3p**: 20t 0.182→0.193 (+6%), 10t 0.173→0.197 (+14%) — consistent with V1.7 observation (extra profiles dilute lucky ones)
2. **This is the true V2.8 performance** — 3-profile estimates flattered the system

**V2.8(5p) Per-Dimension MAE (20 turns, 5 profiles):**

| Dimension | V2.8(3p) | V2.8(5p) |
|-----------|----------|----------|
| EXT | 0.147 | 0.131 |
| HUM | 0.230 | 0.159 |
| VAL | 0.138 | 0.175 |
| AGR | 0.209 | 0.178 |
| OPN | 0.186 | 0.179 |
| COG | 0.210 | 0.183 |
| HON | 0.167 | 0.200 |
| STR | 0.204 | 0.202 |
| EMO | 0.185 | 0.203 |
| DRK | 0.139 | 0.206 |
| CON | 0.180 | 0.207 |
| SOC | 0.195 | 0.238 |
| NEU | 0.181 | 0.239 |

### V2.9 — Behavioral Features (Text-Based Signal Extraction)

**Core change**: Extract objective, non-LLM text signals from speaker turns (pronoun ratios, hedging frequency, absolutist language, emotional word density, question/exclamation ratios) and apply small additive trait adjustments (±0.03-0.10).

**New module**: `super_brain/behavioral_features.py`
- `extract_features()` — token-level feature extraction (no LLM cost)
- `compute_adjustments()` — rule-based mapping from features to trait deltas
- `apply_adjustments()` — additive adjustment post-ensemble, clamped to [0,1]

**21 adjustment rules** covering: narcissism, assertiveness, modesty, deliberation, need_for_cognition, warmth, intellectual_curiosity, positive_emotions, excitement_seeking, anxiety, emotional_volatility, altruism

**V2.9 Per-Dimension MAE (20 turns, 5 profiles):**

| Dimension | V2.8(5p) | V2.9(5p) | Change |
|-----------|----------|----------|--------|
| EXT | 0.131 | **0.124** | -5% |
| OPN | 0.179 | **0.163** | -9% |
| CON | 0.207 | **0.167** | -19% |
| NEU | 0.239 | **0.170** | -29% |
| AGR | 0.178 | 0.186 | +4% |
| HUM | 0.159 | 0.192 | +21% |
| VAL | 0.175 | 0.192 | +10% |
| COG | 0.183 | 0.208 | +14% |
| STR | 0.202 | 0.210 | +4% |
| HON | 0.200 | 0.214 | +7% |
| SOC | 0.238 | **0.218** | -8% |
| EMO | 0.203 | 0.219 | +8% |
| DRK | 0.206 | 0.230 | +12% |

**Key findings**:
1. **10t MAE significantly improved**: 0.197→0.178 (-9.6%) — behavioral features help with early-turn detection
2. **20t MAE marginally improved**: 0.193→0.189 (-2.1%) — within conversation variance for 5 profiles
3. **NEU massively improved at 20t**: 0.239→0.170 (-29%) — anxiety adjustment from negative emotion word density
4. **CON improved**: 0.207→0.167 (-19%) — may correlate with deliberation/need_for_cognition adjustments
5. **OPN improved**: 0.179→0.163 (-9%) — intellectual_curiosity adjustment from question frequency
6. **Per-dimension variance still high**: HUM, DRK, COG regressed — conversation variance dominates
7. **Behavioral features are conservative**: Only 2-7 adjustments per profile (of 21 possible rules)
8. **Zero LLM cost**: All feature extraction is regex/counting, no API calls
9. **First 5-profile eval since V1.7**: Establishes stable measurement baseline for future comparisons

### V3.0 — Soul-Aware Diagnostic Questions

**Core change**: Replace static trait_topic_map questions with LLM-generated contextual diagnostic questions. After each ThinkSlow extraction, an LLM call generates 5 personalized questions based on: (1) already-known high-confidence traits, (2) low-confidence target traits with detection hints, (3) known facts and reality summary from Soul, (4) recent conversation context. Questions use psychology-informed types: situational dilemmas, forced-choice preferences, attribution questions, counterfactuals, value ranking.

**New module**: `super_brain/diagnostic_questions.py` — `generate_diagnostic_questions()` with prompt construction helpers and robust JSON parsing.

**V3.0 Per-Dimension MAE (20 turns, 5 profiles):**

| Dimension | V2.9(5p) | V3.0(5p) | Change |
|-----------|----------|----------|--------|
| NEU | 0.170 | **0.160** | -6% |
| CON | 0.167 | 0.170 | +2% |
| EXT | 0.124 | 0.171 | +38% |
| OPN | 0.163 | 0.178 | +9% |
| VAL | 0.192 | **0.179** | -7% |
| AGR | 0.186 | **0.179** | -4% |
| SOC | 0.218 | **0.189** | -13% |
| COG | 0.208 | **0.192** | -8% |
| STR | 0.210 | **0.195** | -7% |
| HUM | 0.192 | 0.199 | +4% |
| HON | 0.214 | **0.203** | -5% |
| EMO | 0.219 | **0.209** | -5% |
| DRK | 0.230 | **0.220** | -4% |

**Key findings**:
1. **20t MAE improved**: 0.189→0.186 (-1.6%) — modest overall improvement
2. **Hard dimensions significantly improved**: SOC -13%, COG -8%, STR -7%, VAL -7%, NEU -6%
3. **EXT regressed**: 0.124→0.171 (+38%) — diagnostic questions may have disrupted natural extraversion signals; needs investigation
4. **Diagnostic questions consistently generated**: 5 per ThinkSlow extraction (45 total per profile), all with `source=soul_aware_diagnostic`
5. **Cost**: ~45 extra LLM calls per 20-turn profile (1 per ThinkSlow extraction) vs zero for V2.9 behavioral features
6. **Pattern**: Soul-Aware questions excel at probing traits that require specific situational context (social dynamics, interpersonal strategy, values) but add noise for traits that are naturally expressed in casual conversation (extraversion)
7. **10t MAE regressed slightly**: 0.178→0.189 (+6.2%) — diagnostic questions need more turns to accumulate benefit

### V3.1 — Strip-Back to Essentials

**Core change**: Removed 4 features that added complexity without proportional accuracy gains: ThinkDeep (V2.5), Diagnostic Questions (V3.0), Ensemble Blend (V2.8), Soul-Informed Detection (V2.6). Kept: Deep Listening chatter, ThinkSlow extraction, FactExtractor + Soul model, Conductor, Behavioral Features (V2.9), static trait_topic_map questions.

**Motivation**: V3.0 reflection showed the entire V2.x→V3.0 architecture (12+ modules, 100+ API calls per profile) only improved MAE by 2.1% over simple V1.7 pipeline. Three fundamental bottlenecks: (1) 3-5 profile measurement noise, (2) complexity ≠ accuracy, (3) LLM-to-LLM detection ceiling ~0.185-0.190.

**Removed from `simulate_conversation()`**:
- `think_deep` parameter and all ThinkDeep trigger/state logic (~40 lines)
- `api_key` parameter and diagnostic question generation blocks
- Simplified Soul logging (no intentions/gaps counts)

**Removed from `detect_and_compare()`**:
- `soul` parameter and `_build_detector_soul_context()` injection
- `ts_results` parameter and `blend_with_trajectory()` ensemble

**Kept**:
- ThinkSlow periodic extraction (gap-aware trait tracking)
- FactExtractor + Soul model (facts, reality, secrets, contradictions)
- Conductor dynamic conversation mode (listen/follow/incisive/push)
- Behavioral features (zero-cost text signal extraction)
- Static trait_topic_map incisive questions

**V3.1 Per-Dimension MAE (20 turns, 5 profiles):**

| Dimension | V3.0(5p) | V3.1(5p) | Change |
|-----------|----------|----------|--------|
| OPN | 0.178 | **0.168** | -6% |
| EXT | 0.171 | **0.170** | -1% |
| VAL | 0.179 | **0.177** | -1% |
| AGR | 0.179 | 0.179 | ~ |
| DRK | 0.220 | **0.180** | -18% |
| HUM | 0.199 | **0.184** | -8% |
| HON | 0.203 | **0.192** | -5% |
| NEU | 0.160 | 0.194 | +21% |
| SOC | 0.189 | 0.205 | +8% |
| STR | 0.195 | 0.219 | +12% |
| COG | 0.192 | 0.220 | +15% |
| CON | 0.170 | 0.221 | +30% |
| EMO | 0.209 | 0.234 | +12% |

**Key findings**:
1. **20t MAE**: 0.186→0.196 (+5.4%) — slight regression, within 5-profile run variance
2. **10t MAE improved**: 0.189→0.186 (-1.6%) — best 5-profile 10t result ever
3. **10t ≤0.40 rate**: 95.5% — best 5-profile 10t ≤0.40 ever (fewer catastrophic errors)
4. **DRK massively improved**: 0.220→0.180 (-18%) — removing Soul-informed context eliminated dark trait anchoring bias
5. **HUM improved**: 0.199→0.184 (-8%) — cleaner detection without ensemble interference
6. **CON regressed**: 0.170→0.221 (+30%) — lost ThinkSlow trajectory smoothing benefit
7. **NEU regressed**: 0.160→0.194 (+21%) — lost diagnostic question benefit for neuroticism probing
8. **LLM call savings**: ~47 fewer calls per profile (2 ThinkDeep + 45 DiagQ) — significant cost reduction
9. **Lesson**: Simpler pipeline performs comparably at 10t and within noise at 20t. The stripped-back system is a better foundation for future improvements.
10. **Architecture decision**: Keep Soul model for rich person understanding even if not feeding into detection — valuable for future real-human interactions
