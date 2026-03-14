import re

from models import Turn

_WORD_RE = re.compile(r"[a-zA-Z']+")

_HEDGING_PHRASES = re.compile(
    r"\b(?:I think|maybe|perhaps|it seems|kind of|sort of|might be|could be|"
    r"I believe|in my opinion|I guess|probably|possibly|arguably|"
    r"I would say|it appears|to some extent|in a way|more or less)\b",
    re.IGNORECASE,
)

_FILLER_WORDS = re.compile(
    r"\b(?:like|you know|I mean|basically|literally|right|um|uh|well|"
    r"so yeah|okay so|anyway|whatever)\b",
    re.IGNORECASE,
)

MIN_WORDS = 20


def score_hedging_readability(
    turns: list[Turn],
    debaters: list[str] | None = None,
) -> None:
    """Compute discourse quality from hedging, fillers, and vocabulary diversity."""
    for turn in turns:
        if debaters and turn.speaker not in debaters:
            continue

        words = turn.text.split()
        word_count = len(words)
        if word_count < MIN_WORDS:
            continue

        # Hedging rate
        hedging_count = len(_HEDGING_PHRASES.findall(turn.text))
        hedging_rate = hedging_count / word_count

        # Filler rate
        filler_count = len(_FILLER_WORDS.findall(turn.text))
        filler_rate = filler_count / word_count

        # Vocabulary diversity (type-token ratio)
        tokens = _WORD_RE.findall(turn.text.lower())
        unique_tokens = set(tokens)
        vocab_diversity = len(unique_tokens) / len(tokens) if tokens else 0.0

        # Discourse quality: combine components, each capped at [0, 1]
        vocab_component = max(min(vocab_diversity, 1.0), 0.0)
        hedge_component = max(min(1 - hedging_rate * 10, 1.0), 0.0)
        filler_component = max(min(1 - filler_rate * 10, 1.0), 0.0)

        turn.discourse_quality = round(
            vocab_component * 0.4 + hedge_component * 0.3 + filler_component * 0.3,
            4,
        )
