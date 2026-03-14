import re

import numpy as np
from models import Turn

_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')

_REBUTTAL_STARTS = re.compile(
    r"^(?:no[,.]|that(?:'s| is) (?:wrong|not true|incorrect|false|not how)|"
    r"you(?:'re| are) (?:wrong|mistaken)|actually,|incorrect)",
    re.IGNORECASE,
)

PREMISE_INDICATORS = re.compile(
    r"\b(?:because|since|given that|due to|owing to|as a result of|"
    r"on the grounds that|the reason is|based on|for example|for instance|"
    r"such as|according to|as shown by|as evidenced by|as demonstrated by|"
    r"which shows|which demonstrates|this is evidenced by|the data shows|"
    r"research shows|studies show|evidence shows)\b",
    re.IGNORECASE,
)

# Concrete element detectors for reason quality check
_RE_NUMBER = re.compile(r'\b\d[\d,]*(?:\.\d+)?(?:\s*(?:%|percent|million|billion|thousand|kg|km|miles|degrees|years))?\b')
_RE_NAMED_ENTITY = re.compile(r'(?<![.!?]\s)(?<!\A)(?<!\n)\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b')
_RE_TECHNICAL = re.compile(r'\b[a-zA-Z]{9,}\b')  # words >8 chars
_RE_EVIDENCE_PHRASE = re.compile(
    r"\b(?:according to|research shows|studies show|data shows|"
    r"for example|for instance|evidence shows|experiment|"
    r"measured|observed|documented|recorded|published)\b",
    re.IGNORECASE,
)

# Abstract non-reasons to reject
_ABSTRACT_REASONS = re.compile(
    r"\b(?:that(?:'s| is) (?:how it works|obvious|just how it is|common sense)|"
    r"everyone knows|it(?:'s| is) obvious|it(?:'s| is) clear|"
    r"it(?:'s| is) just|you can see|just look|just think)\b",
    re.IGNORECASE,
)

# Attack words from ad_hominem — premises containing these with few content words
# are insults disguised as reasoning
_ATTACK_WORDS = {
    "stupid", "idiot", "fool", "ignorant", "incompetent", "liar", "dishonest",
    "clueless", "delusional", "hypocrite", "corrupt", "pathetic", "coward",
    "moron", "dumb", "naive", "arrogant", "selfish", "lazy",
    "disgrace", "shameful", "unqualified", "biased", "bigot", "racist",
    "sexist", "extremist", "radical", "fanatic", "hack", "fraud", "phony",
    "crook", "criminal", "deplorable", "disgusting", "despicable",
    "suck", "sucks", "terrible", "horrible", "awful", "ridiculous", "absurd",
}

# Stopwords for content word counting
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "must", "need",
    "i", "me", "my", "we", "us", "our", "you", "your", "he", "him", "his",
    "she", "her", "it", "its", "they", "them", "their", "that", "this",
    "what", "which", "who", "whom", "how", "when", "where", "why",
    "not", "no", "nor", "if", "then", "than", "too", "very", "just",
    "but", "and", "or", "so", "as", "at", "by", "for", "in", "of", "on",
    "to", "up", "out", "off", "with", "from", "into", "about", "all",
    "also", "more", "some", "any", "each", "only", "own", "same", "get",
    "got", "say", "said", "says", "go", "goes", "went", "come", "came",
    "make", "made", "take", "took", "see", "saw", "know", "knew",
    "think", "thought", "here", "there", "now", "then",
}

# Filler words to ignore in content counting
_FILLERS = {
    "uh", "um", "like", "right", "okay", "ok", "well", "basically",
    "actually", "literally", "really", "pretty", "honestly", "obviously",
    "clearly", "anyway", "anyways", "yeah", "yes", "no", "hey", "look",
}

MIN_WORDS = 15
MIN_COUPLING = 0.25  # claim-reason must be at least this similar
MIN_REASON_CONTENT_WORDS = 2  # minimum content words in reason
MAX_ATTACK_CONTENT_WORDS = 5  # if attack word + fewer content words = insult


def _count_content_words(text: str) -> int:
    """Count substantive content words (non-stopword, non-filler, >3 chars)."""
    return sum(
        1 for w in text.lower().split()
        if len(w) > 3 and w not in _STOPWORDS and w not in _FILLERS
    )


def _is_insult_premise(text: str) -> bool:
    """Check if a premise is an insult disguised as reasoning."""
    text_lower = text.lower()
    has_attack = any(w in text_lower.split() for w in _ATTACK_WORDS)
    if not has_attack:
        return False
    content_words = _count_content_words(text)
    return content_words < MAX_ATTACK_CONTENT_WORDS


def _has_concrete_element(text: str) -> bool:
    """Check if text contains at least one concrete element: number, named entity,
    technical term, or evidence phrase."""
    if _RE_NUMBER.search(text):
        return True
    if _RE_NAMED_ENTITY.search(text):
        return True
    if _RE_TECHNICAL.search(text):
        return True
    if _RE_EVIDENCE_PHRASE.search(text):
        return True
    return False


def score_premise_sufficiency(
    turns: list[Turn],
    embedder,
    debaters: list[str] | None = None,
) -> None:
    """Measure whether a speaker provides semantically coupled AND concrete reasons.

    If speech_act exists, counts turns with speech_act == "explanation" or "correction"
    as turns with premises. Falls back to regex if no speech_act data.

    Filters out:
    - Reasons with fewer than 2 content words (non-filler, non-stopword, >3 chars)
    - Insults disguised as reasoning (attack words + few content words)
    - Purely abstract reasons ("that's how it works", "everyone knows")
    """
    # Check if structure extraction data is available
    has_structure = any(t.speech_act is not None for t in turns)

    if has_structure:
        # Use speech_act for premise detection
        for turn in turns:
            if debaters and turn.speaker not in debaters:
                continue
            if turn.speech_act is None:
                continue
            if turn.speech_act in ("explanation", "correction"):
                turn.premise_sufficiency = 1.0
            elif turn.speech_act in ("claim", "rebuttal", "challenge"):
                turn.premise_sufficiency = 0.0
            # dismissal/insult/agreement/concession/fragment/backchannel: leave as None
        return

    for turn in turns:
        if debaters and turn.speaker not in debaters:
            continue

        if len(turn.text.split()) < MIN_WORDS:
            continue

        sentences = _SENTENCE_SPLIT.split(turn.text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        # Count claim sentences: assertive, not questions, not pure rebuttals
        claim_count = 0
        for s in sentences:
            if s.endswith("?"):
                continue
            if _REBUTTAL_STARTS.match(s):
                continue
            claim_count += 1

        # Find all premise indicators per sentence and extract claim-reason pairs
        total_premises = 0
        coupled_premises = 0

        for si, sentence in enumerate(sentences):
            for m in PREMISE_INDICATORS.finditer(sentence):
                claim_text = sentence[:m.start()].strip()
                reason_text = sentence[m.end():].strip()

                # Strip trailing punctuation from reason if it runs past sentence
                reason_match = re.match(r'^(.*?[.!?])', reason_text, re.DOTALL)
                if reason_match:
                    reason_text = reason_match.group(1).strip()

                # If claim part is too short, use previous sentence as the claim
                if len(claim_text.split()) < 3 and si > 0:
                    claim_text = sentences[si - 1].strip()

                # Need substance in both halves (total words)
                if len(claim_text.split()) < 3 or len(reason_text.split()) < 3:
                    continue

                # Reason must have enough CONTENT words (not fillers/stopwords)
                if _count_content_words(reason_text) < MIN_REASON_CONTENT_WORDS:
                    continue

                # Reject insults disguised as reasoning
                if _is_insult_premise(reason_text):
                    continue

                # Reject purely abstract reasons
                if _ABSTRACT_REASONS.search(reason_text) and not _has_concrete_element(reason_text):
                    continue

                total_premises += 1

                # Embed both halves and check semantic coupling
                claim_emb = embedder.embed(claim_text)
                reason_emb = embedder.embed(reason_text)
                similarity = float(np.dot(claim_emb, reason_emb))

                # Must have both coupling AND concrete content
                if similarity > MIN_COUPLING and _has_concrete_element(reason_text):
                    coupled_premises += 1

                break  # one premise per sentence

        if total_premises == 0:
            continue

        coupling_rate = coupled_premises / total_premises
        denom = max(claim_count, 1)
        turn.premise_sufficiency = round(
            min((total_premises / denom) * coupling_rate, 1.0), 4
        )
