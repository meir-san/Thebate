import re
from models import Turn
import config

FILLER_RE = re.compile(
    r"^(yeah|yep|no|yes|okay|ok|right|exactly|sure|absolutely|"
    r"i mean|you know|like i said|so|well|um|uh|hmm|wait|look|listen)\b",
    re.IGNORECASE
)


def is_claim(sentence: str) -> bool:
    """A claim is a declarative sentence with enough content to evaluate."""
    s = sentence.strip()
    if len(s.split()) < 6:
        return False
    if s.endswith('?'):
        return False
    if FILLER_RE.match(s):
        return False
    return True


def is_supported(sentence: str) -> bool:
    """A supported claim contains at least one reasoning connector."""
    s_lower = sentence.lower()
    return any(connector in s_lower for connector in config.REASONING_CONNECTORS)


def score_claims(turns: list[Turn]) -> None:
    """Mutates turns in place. Sets turn.claim_count and turn.supported_claim_count."""
    for turn in turns:
        # Split on sentence-ending punctuation
        sentences = [s.strip() for s in re.split(r'[.!?]+', turn.text) if s.strip()]
        claims = [s for s in sentences if is_claim(s)]
        supported = [s for s in claims if is_supported(s)]
        turn.claim_count = len(claims)
        turn.supported_claim_count = len(supported)
