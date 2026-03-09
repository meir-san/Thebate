from dataclasses import dataclass, field, asdict
from typing import Literal

FlagType = Literal["dodge", "low_engagement", "topic_drift", "unsupported_claim"]


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
    flags: list[Flag] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["flags"] = [f.to_dict() for f in self.flags]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Turn":
        flags = [Flag.from_dict(f) for f in d.pop("flags", [])]
        return cls(**d, flags=flags)


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
    overall_score: float            # 0–100

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SpeakerStats":
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
