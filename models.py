from dataclasses import dataclass, field, asdict
from typing import Literal

FlagType = Literal[
    "dodge", "low_engagement", "topic_drift", "unsupported_claim",
    "correction", "position_shift", "low_evidence",
    "ad_hominem", "strawman", "whataboutism", "red_herring",
    "gish_gallop", "circular_reasoning", "false_dichotomy",
]


@dataclass
class Flag:
    turn_index: int
    flag_type: FlagType
    score: float        # raw score that triggered the flag
    threshold: float    # threshold it crossed
    explanation: str    # e.g. "Response similarity to question: 0.14, threshold: 0.25"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Flag":
        return cls(**d)


@dataclass
class Turn:
    index: int
    speaker: str        # "SPEAKER_00" until phase1 renames, then real name
    text: str
    start_ms: int       # milliseconds from video start
    end_ms: int
    # Set by phase2 — default None means "not yet scored" for engagement/drift
    engagement_score: float | None = None   # None = skipped (short turn)
    is_dodge: bool = False
    claim_count: int = 0
    supported_claim_count: int = 0
    topic_drift_score: float | None = None  # None = skipped (short turn)
    evidence_markers: int = 0
    evidence_density: float = 0.0
    ad_hominem_count: int = 0
    strawman_detected: bool = False
    whataboutism_detected: bool = False
    red_herring_detected: bool = False
    gish_gallop_detected: bool = False
    circular_reasoning_detected: bool = False
    false_dichotomy_detected: bool = False
    opponent_term_adoption: float | None = None
    targeting_score: float | None = None
    schemes: list[str] = field(default_factory=list)
    scheme_diversity: float = 0.0
    paraphrase_fidelity: float | None = None
    engagement_quality_level: int | None = None
    flags: list[Flag] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["flags"] = [f.to_dict() for f in self.flags]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Turn":
        flags = [Flag.from_dict(f) for f in d.pop("flags", [])]
        schemes = d.pop("schemes", [])
        d.setdefault("ad_hominem_count", 0)
        d.setdefault("strawman_detected", False)
        d.setdefault("whataboutism_detected", False)
        d.setdefault("red_herring_detected", False)
        d.setdefault("gish_gallop_detected", False)
        d.setdefault("circular_reasoning_detected", False)
        d.setdefault("false_dichotomy_detected", False)
        d.setdefault("opponent_term_adoption", None)
        d.setdefault("targeting_score", None)
        d.setdefault("scheme_diversity", 0.0)
        d.setdefault("paraphrase_fidelity", None)
        d.setdefault("engagement_quality_level", None)
        return cls(**d, flags=flags, schemes=schemes)


@dataclass
class SpeakerStats:
    speaker: str
    turn_count: int
    avg_engagement: float
    total_dodges: int
    questions_faced: int
    dodge_rate: float               # 0.0 if questions_faced == 0
    total_claims: int
    supported_claims: int
    claim_support_ratio: float      # 1.0 if total_claims == 0
    avg_topic_drift: float
    corrections_received: int
    corrections_acknowledged: int
    correction_absorption_rate: float   # higher = better
    position_shifts: int
    consistency_score: float            # higher = better
    concessions_made: int
    concessions_engaged: int            # followed by on-topic continuation
    concessions_pivot: int              # followed by topic change
    concession_rate: float              # higher = better (engaged only)
    avg_evidence_density: float         # higher = better
    total_evidence_markers: int
    fallacy_rate: float = 0.0           # fallacy flags / total turns
    fallacy_counts: dict = field(default_factory=dict)  # per-type counts
    avg_opponent_term_adoption: float = 0.0
    avg_targeting_score: float = 0.0
    scheme_diversity: float = 0.0
    avg_paraphrase_fidelity: float = 0.0
    avg_engagement_quality: float = 0.0
    overall_score: float = 0.0         # 0–100

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SpeakerStats":
        # Backward compat: fill in defaults for new fields missing from old JSON
        d.setdefault("corrections_received", 0)
        d.setdefault("corrections_acknowledged", 0)
        d.setdefault("correction_absorption_rate", 1.0)
        d.setdefault("position_shifts", 0)
        d.setdefault("consistency_score", 1.0)
        d.setdefault("concessions_made", 0)
        d.setdefault("concessions_engaged", 0)
        d.setdefault("concessions_pivot", 0)
        d.setdefault("concession_rate", 0.0)
        d.setdefault("avg_evidence_density", 0.0)
        d.setdefault("total_evidence_markers", 0)
        d.setdefault("fallacy_rate", 0.0)
        d.setdefault("fallacy_counts", {})
        d.setdefault("avg_opponent_term_adoption", 0.0)
        d.setdefault("avg_targeting_score", 0.0)
        d.setdefault("scheme_diversity", 0.0)
        d.setdefault("avg_paraphrase_fidelity", 0.0)
        d.setdefault("avg_engagement_quality", 0.0)
        d.setdefault("overall_score", 0.0)
        return cls(**d)


@dataclass
class DebateResult:
    title: str
    youtube_url: str
    topic: str
    duration_ms: int
    speakers: list[str]                     # in order of first appearance
    debaters: list[str]                     # subset of speakers who are actually being scored
    turns: list[Turn]
    stats: dict[str, SpeakerStats]          # keyed by speaker name, empty until phase2
    generated_at: str                       # ISO 8601

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "youtube_url": self.youtube_url,
            "topic": self.topic,
            "duration_ms": self.duration_ms,
            "speakers": self.speakers,
            "debaters": self.debaters,
            "turns": [t.to_dict() for t in self.turns],
            "stats": {k: v.to_dict() for k, v in self.stats.items()},
            "generated_at": self.generated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DebateResult":
        turns = [Turn.from_dict(t) for t in d["turns"]]
        stats = {k: SpeakerStats.from_dict(v) for k, v in d.get("stats", {}).items()}
        return cls(
            title=d["title"],
            youtube_url=d["youtube_url"],
            topic=d["topic"],
            duration_ms=d["duration_ms"],
            speakers=d["speakers"],
            debaters=d.get("debaters", d["speakers"]),
            turns=turns,
            stats=stats,
            generated_at=d["generated_at"],
        )
