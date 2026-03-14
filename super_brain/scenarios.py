"""Scenario Engine for trait-activated personality detection (V4.0).

Each scenario is a focused 3-4 turn conversation designed to elicit specific
personality traits. Based on Trait Activation Theory (Tett & Burnett 2003).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Scenario:
    """A conversation scenario designed to activate specific traits."""
    id: str
    description: str
    chatter_setup: str  # What Chatter introduces to trigger the scenario
    target_traits: list[str]  # Traits this scenario is designed to elicit
    turns: int = 3  # How many exchange rounds for this scenario


# ── 16 Scene Packs covering all 69 traits ──────────────────────────────────

SCENARIOS: list[Scenario] = [
    Scenario(
        id="social_trust",
        description="信任陌生人、借钱给朋友",
        chatter_setup=(
            "Start by sharing that someone you met online two weeks ago wants to go into "
            "business together, and ask if that sounds reasonable. Then ask if the other "
            "person has ever been cheated by a friend — maybe about lending money."
        ),
        target_traits=["trust", "straightforwardness", "sincerity", "fairness"],
        turns=3,
    ),
    Scenario(
        id="helping_others",
        description="路上看到流浪汉，不太熟的同事要帮忙搬家",
        chatter_setup=(
            "Share that you saw a homeless person digging through trash for food on your "
            "way home, and ask how the other person would react. Then mention that a "
            "coworker you barely know asked you to help them move this weekend."
        ),
        target_traits=["altruism", "tender_mindedness", "care_harm", "empathy_affective", "compliance"],
        turns=3,
    ),
    Scenario(
        id="conflict_scenario",
        description="室友空调温度争吵，会议上挑战领导",
        chatter_setup=(
            "Describe a fight with your roommate over AC temperature (you want 26°C, they "
            "want 22°C) and ask how they'd handle it. Then describe a meeting where the "
            "boss proposed a plan with obvious flaws but everyone is nodding along."
        ),
        target_traits=["compliance", "assertiveness", "conflict_assertiveness",
                       "conflict_cooperativeness", "social_dominance", "angry_hostility"],
        turns=4,
    ),
    Scenario(
        id="imagination_creativity",
        description="变成另一个人的幻想，设计自己的世界",
        chatter_setup=(
            "Ask: 'If you woke up tomorrow as a completely different person, who would you "
            "want to be?' Then ask: 'If you could design your own world from scratch, what "
            "would it look like?' Explore their ideas with genuine curiosity."
        ),
        target_traits=["fantasy", "aesthetics", "ideas", "curiosity",
                       "need_for_cognition", "cognitive_flexibility"],
        turns=3,
    ),
    Scenario(
        id="stress_pressure",
        description="等待重要结果，多重压力同时来",
        chatter_setup=(
            "Ask about waiting for an important result (like a promotion decision) — what "
            "state are they in during the wait? Then pose a hypothetical: work problems + "
            "friend conflict + feeling unwell all hit at once. Can they handle it?"
        ),
        target_traits=["anxiety", "vulnerability", "emotional_regulation",
                       "emotional_volatility", "depression", "optimism"],
        turns=4,
    ),
    Scenario(
        id="achievement_work",
        description="被夸项目做得好，讨论努力有没有用",
        chatter_setup=(
            "Compliment them — say you heard their recent project was praised by leadership. "
            "Observe their response. Then ask a deeper question: 'Do you think hard work "
            "really pays off? Sometimes it feels like results are the same no matter what.'"
        ),
        target_traits=["modesty", "competence", "achievement_striving",
                       "self_discipline", "humility_hexaco", "narcissism"],
        turns=3,
    ),
    Scenario(
        id="money_values",
        description="突然有大钱会买什么，赚钱vs做喜欢的事",
        chatter_setup=(
            "Ask: 'If you suddenly got a huge amount of money, what's the first thing you'd "
            "buy?' Then ask: 'If you had to choose — making money or doing what you love — "
            "and couldn't have both, which would you pick?'"
        ),
        target_traits=["greed_avoidance", "values_openness", "locus_of_control",
                       "authority_respect", "fairness_justice"],
        turns=3,
    ),
    Scenario(
        id="social_embarrassment",
        description="公共场合摔跤，发帖没人赞",
        chatter_setup=(
            "Ask: 'If you tripped and fell in a crowded place with everyone watching, how "
            "would you react?' Then ask about posting something online and getting zero "
            "likes after two hours — how would that feel?"
        ),
        target_traits=["self_consciousness", "attachment_anxiety", "self_mythologizing",
                       "warmth", "gregariousness"],
        turns=3,
    ),
    Scenario(
        id="impulsive_decisions",
        description="限时折扣要不要买，事后后悔的冲动决定",
        chatter_setup=(
            "Describe seeing an amazing limited-time deal on something you don't really need "
            "and ask if they'd buy it. Then ask: 'Have you ever made an impulsive decision "
            "that you really regretted afterward?'"
        ),
        target_traits=["impulsiveness", "decisiveness", "deliberation",
                       "order", "intuitive_vs_analytical"],
        turns=3,
    ),
    Scenario(
        id="new_experiences",
        description="周末多假期做什么，朋友推荐没试过的活动",
        chatter_setup=(
            "Ask: 'If you suddenly had two extra days off this weekend with zero plans, "
            "what would you do?' Then say a friend recommended trying something totally "
            "new — like rock climbing or improv comedy — would they go?"
        ),
        target_traits=["actions", "excitement_seeking", "activity_level",
                       "positive_emotions", "curiosity"],
        turns=3,
    ),
    Scenario(
        id="honesty_dilemma",
        description="朋友蛋糕味道一般怎么评价，面试包装自己",
        chatter_setup=(
            "Share a dilemma: your friend baked a cake that tastes mediocre but is eagerly "
            "awaiting your review. What would you say? Then ask: 'In a job interview, do "
            "you present yourself as better than you really are?'"
        ),
        target_traits=["straightforwardness", "sincerity", "machiavellianism",
                       "information_control", "politeness"],
        turns=3,
    ),
    Scenario(
        id="humor_situations",
        description="尴尬经历怎么化解，窘境中的反应",
        chatter_setup=(
            "Share a funny embarrassing story of your own (like accidentally calling your "
            "teacher 'mom'). Then ask how they usually react when something embarrassing "
            "happens — do they laugh it off, feel mortified, make fun of themselves?"
        ),
        target_traits=["humor_affiliative", "humor_self_enhancing", "humor_aggressive",
                       "humor_self_defeating", "charm_influence"],
        turns=3,
    ),
    Scenario(
        id="relationships_attachment",
        description="伴侣/好友突然冷淡，亲密关系中的不安全感",
        chatter_setup=(
            "Share that you've noticed your close friend/partner has been acting distant "
            "lately and you're not sure why. Ask how they'd feel and what they'd do in "
            "that situation. Explore their relationship patterns."
        ),
        target_traits=["attachment_anxiety", "attachment_avoidance",
                       "hot_cold_oscillation", "mirroring_ability", "empathy_cognitive"],
        turns=4,
    ),
    Scenario(
        id="moral_dilemma",
        description="考试偷看答案，游戏bug刷金币",
        chatter_setup=(
            "Pose a moral dilemma: 'If you were certain you wouldn't get caught, would "
            "you peek at someone's answers during an exam?' Then: 'What about a game bug "
            "that lets you farm unlimited gold — would you use it?'"
        ),
        target_traits=["fairness", "psychopathy", "sadism", "dutifulness", "loyalty_group"],
        turns=3,
    ),
    Scenario(
        id="emotions_feelings",
        description="情绪起伏大不大，好朋友倾诉困难",
        chatter_setup=(
            "Ask if they consider themselves emotionally expressive or more even-keeled. "
            "Then share that a close friend came to them saying life has been really hard "
            "lately. How would they respond?"
        ),
        target_traits=["feelings", "empathy_affective", "empathy_cognitive",
                       "tender_mindedness", "emotional_regulation"],
        turns=3,
    ),
    Scenario(
        id="life_philosophy",
        description="天生有人更优秀吗，运气vs能力",
        chatter_setup=(
            "Ask: 'Do you think some people are just naturally better or more talented than "
            "others?' Then: 'Compared to most people, would you say you've been more lucky "
            "or more capable?'"
        ),
        target_traits=["humility_hexaco", "narcissism", "locus_of_control",
                       "authority_respect", "verbosity"],
        turns=3,
    ),
]


def get_coverage_matrix() -> dict[str, list[str]]:
    """Return {trait_name: [scenario_ids]} showing which scenarios cover each trait."""
    coverage: dict[str, list[str]] = {}
    for s in SCENARIOS:
        for t in s.target_traits:
            coverage.setdefault(t, []).append(s.id)
    return coverage


def get_scenario_sequence(seed: int = 0) -> list[Scenario]:
    """Return a shuffled sequence of all scenarios for one eval run.

    Different seeds produce different orderings to avoid order effects.
    """
    import random
    rng = random.Random(seed)
    seq = list(SCENARIOS)
    rng.shuffle(seq)
    return seq
