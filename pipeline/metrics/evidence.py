import re
from models import Turn, Flag
import config

# === EMPIRICAL evidence markers (weight: 1.0 each) ===
# These represent actual data, measurements, experiments, or verifiable facts.

_RE_YEAR = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
_RE_PERCENT = re.compile(r"\b\d+(\.\d+)?%")
_RE_STAT_NUMBER = re.compile(
    r"\b\d+[\.,]?\d*\s*(million|billion|thousand|people|deaths|casualties|years|months|weeks|days)\b",
    re.IGNORECASE,
)
_RE_DIRECT_QUOTE = re.compile(r'["\u201c]([^"\u201d]{12,})["\u201d]')

_RE_SCIENTIFIC = re.compile(
    r"\b(?:the\s+(?:theory|law|laws)\s+of|experiment|observation|hypothesis|"
    r"peer[\s-]reviewed|published\s+in|the\s+study|"
    # Named theories/laws
    r"Newton(?:'s)?\s+law|general\s+relativity|special\s+relativity|"
    r"universal\s+gravitation|theory\s+of\s+evolution|thermodynamics|"
    r"quantum\s+mechanics|Kepler(?:'s)?\s+law|"
    # Mechanism language (spoken science)
    r"the\s+way\s+it\s+works\s+is|what\s+happens\s+is|the\s+process\s+of|"
    r"this\s+is\s+how|the\s+mechanism\s+is|the\s+mechanism\s+for|"
    # Calculation/measurement language
    r"you\s+can\s+calculate|you\s+can\s+measure|the\s+equation|the\s+formula|"
    r"parts\s+per\s+million|degrees\s+celsius|millimeters\s+per|"
    # Observational evidence
    r"has\s+been\s+observed|has\s+been\s+measured|has\s+been\s+replicated|"
    r"we\s+have\s+observed|you\s+can\s+observe|you\s+can\s+see)\b",
    re.IGNORECASE,
)

_RE_LOGICAL = re.compile(
    r"\b(?:if\b.{1,40}\bthen\b|it\s+follows\s+that|this\s+contradicts|"
    r"this\s+is\s+inconsistent\s+with|by\s+that\s+logic|"
    r"the\s+implication\s+is|logically)\b",
    re.IGNORECASE,
)

_RE_EMPIRICAL = re.compile(
    r"\b(?:you\s+can\s+(?:observe|measure|test|verify)|"
    r"we\s+can\s+(?:see|verify|observe|measure|test)|"
    r"the\s+(?:data|evidence)\s+(?:shows|suggests|indicates|demonstrates)|"
    r"demonstrably|empirically|measurably)\b",
    re.IGNORECASE,
)

_RE_METHODOLOGICAL = re.compile(
    r"\b(?:controlled\s+experiment|sample\s+size|replication|falsifiable|"
    r"peer\s+review|scientific\s+method|burden\s+of\s+proof)\b",
    re.IGNORECASE,
)

# === AUTHORITY citation markers (weight: 0.3 each) ===
# Named sources without accompanying data — weak evidence on their own.

_RE_NAMED_SOURCE = re.compile(
    r"(?:according to|per|from|citing|says|reported by|published by|based on)\s+"
    r"([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)",
)

_RE_NAMED_CONCEPT = re.compile(
    r"\b(?:the\s+(?:principle|concept|model|theory|law)\s+of)\s+"
    r"([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)",
)

# Named scientists mentioned as authority (spoken debate pattern)
_RE_NAMED_SCIENTIST = re.compile(
    r"\b(?:Einstein|Newton|Galileo|Copernicus|Darwin|Kepler|Hawking|"
    r"Ptolemy|Eratosthenes|Archimedes|Faraday|Maxwell|Bohr|"
    r"Feynman|Curie|Planck|Hubble)\b",
)

EMPIRICAL_WEIGHT = 1.0
AUTHORITY_WEIGHT = 0.3
EVIDENCE_SATURATION = 3  # weighted markers summing to 3+ score 1.0


def _count_evidence_markers(text: str) -> tuple[float, int]:
    """Count evidence markers with weighting. Returns (weighted_score, raw_count)."""
    empirical = 0
    empirical += len(_RE_YEAR.findall(text))
    empirical += len(_RE_PERCENT.findall(text))
    empirical += len(_RE_STAT_NUMBER.findall(text))
    empirical += len(_RE_DIRECT_QUOTE.findall(text))
    empirical += len(_RE_SCIENTIFIC.findall(text))
    empirical += len(_RE_LOGICAL.findall(text))
    empirical += len(_RE_EMPIRICAL.findall(text))
    empirical += len(_RE_METHODOLOGICAL.findall(text))

    authority = 0
    authority += len(_RE_NAMED_SOURCE.findall(text))
    authority += len(_RE_NAMED_CONCEPT.findall(text))
    authority += len(_RE_NAMED_SCIENTIST.findall(text))

    weighted = empirical * EMPIRICAL_WEIGHT + authority * AUTHORITY_WEIGHT
    raw_count = empirical + authority
    return weighted, raw_count


def score_evidence(
    turns: list[Turn],
    debaters: list[str] | None = None,
) -> None:
    """Mutates turns in place. Scores evidence by weighted marker count per turn.
    Empirical evidence (data, experiments, measurements) weights 1.0.
    Authority citations (named sources without data) weight 0.3.
    A turn with weighted markers summing to 3+ scores 1.0.

    When proposition exists, searches BOTH proposition and raw text for markers
    (proposition is clean but raw text has specific numbers and names)."""
    min_words = config.MIN_WORDS_EVIDENCE

    for turn in turns:
        if debaters and turn.speaker not in debaters:
            continue

        word_count = len(turn.text.split())

        # Search both proposition and raw text, take the best result
        weighted_raw, count_raw = _count_evidence_markers(turn.text)
        if turn.proposition:
            weighted_prop, count_prop = _count_evidence_markers(turn.proposition)
            # Combine: unique markers from both sources (use max as approximation)
            weighted = max(weighted_raw, weighted_prop + weighted_raw * 0.5)
            raw_count = max(count_raw, count_prop + count_raw)
        else:
            weighted = weighted_raw
            raw_count = count_raw

        turn.evidence_markers = raw_count
        turn.evidence_density = round(min(weighted / EVIDENCE_SATURATION, 1.0), 4)

        if raw_count == 0 and word_count > min_words:
            turn.flags.append(Flag(
                turn_index=turn.index,
                flag_type="low_evidence",
                score=0.0,
                threshold=0.0,
                explanation=f"No evidence markers in {word_count}-word turn",
            ))
