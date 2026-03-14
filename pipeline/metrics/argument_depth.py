import re

import numpy as np
from models import Turn

_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')

PREMISE_INDICATORS = re.compile(
    r"\b(?:because|since|given that|due to|owing to|as a result of|"
    r"on the grounds that|the reason is)\b",
    re.IGNORECASE,
)

CONCLUSION_INDICATORS = re.compile(
    r"\b(?:therefore|thus|hence|consequently|it follows that|"
    r"which means|this proves|this shows|this demonstrates|"
    r"that's why|which is why)\b",
    re.IGNORECASE,
)

EVIDENCE_INDICATORS = re.compile(
    r"\b(?:for example|for instance|such as|specifically|as shown by|"
    r"according to|research shows|studies show|the data shows|"
    r"historically|empirically)\b",
    re.IGNORECASE,
)

# Filler words/phrases to strip before chain analysis
_FILLER_WORDS = re.compile(
    r"\b(?:uh|um|uh+m*|like|you know|I mean|sort of|kind of|right|okay|ok|"
    r"basically|actually|literally|really|just|pretty much|"
    r"honestly|frankly|obviously|clearly|of course|anyway|anyways|"
    r"look|hey|yeah|yes|ah|oh)\b[,.]?\s*",
    re.IGNORECASE,
)
_REPEATED_WORDS = re.compile(r"\b(\w+)(?:\s+\1){2,}\b", re.IGNORECASE)
_FALSE_STARTS = re.compile(r"\b(\w+(?:\s+\w+){0,3})\s+\1\b", re.IGNORECASE)

# Stopwords for content word extraction
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "must", "need",
    "i", "me", "my", "we", "us", "our", "you", "your", "he", "him", "his",
    "she", "her", "it", "its", "they", "them", "their", "that", "this",
    "not", "no", "but", "and", "or", "so", "as", "at", "by", "for", "in",
    "of", "on", "to", "up", "out", "off", "with", "from", "into", "about",
    "all", "also", "more", "some", "any", "each", "only", "own", "same",
    "get", "got", "say", "said", "go", "went", "come", "came",
    "make", "made", "take", "took", "see", "saw", "know", "knew",
    "think", "thought", "here", "there", "now", "then",
}

MIN_WORDS = 20
MAX_CHAIN = 5  # normalize by this — a chain of 5 explicit steps is excellent
CHAIN_SIM_MIN = 0.15    # below this = topic change (broken chain)
CHAIN_SIM_MAX = 0.6     # above this = restating same point (padding, not depth)
IMPLICIT_CONTINUATION_MIN = 0.2  # sim-only threshold for implicit topical continuation
CONCLUSION_MULTIPLIER = 1.5  # bonus for completed chains
MIN_SHARED_CONTENT_WORDS = 1  # shared content words to continue chain without connector


def _has_connector(sentence: str) -> bool:
    return bool(
        PREMISE_INDICATORS.search(sentence)
        or CONCLUSION_INDICATORS.search(sentence)
        or EVIDENCE_INDICATORS.search(sentence)
    )


def _has_conclusion(sentence: str) -> bool:
    return bool(CONCLUSION_INDICATORS.search(sentence))


def _clean_text_for_chains(text: str) -> str:
    """Remove filler words and speech disfluencies before chain analysis."""
    cleaned = text
    for _ in range(5):
        prev = cleaned
        cleaned = _FILLER_WORDS.sub(" ", cleaned)
        if cleaned == prev:
            break
    cleaned = _FALSE_STARTS.sub(r"\1", cleaned)
    cleaned = _REPEATED_WORDS.sub(r"\1", cleaned)
    cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip()
    return cleaned


def _get_content_words(sentence: str) -> set[str]:
    """Extract content words (non-stopword, >3 chars) from a sentence."""
    return {
        w.lower() for w in sentence.split()
        if len(w) > 3 and w.lower() not in _STOPWORDS
    }


def score_argument_depth(
    turns: list[Turn],
    turn_embeddings: dict[int, np.ndarray],
    embedder,
    debaters: list[str] | None = None,
) -> None:
    """Measure reasoning chain length per turn.

    Chain continues if:
    1. Sentence has a connector AND similarity is in the Goldilocks zone (0.15-0.6), OR
    2. Sentence shares 2+ content words with the previous sentence (catches implicit
       links like "The shadow angle was different. This difference proves curvature.")

    Text is pre-cleaned to remove filler words that break chains.
    """
    for turn in turns:
        if debaters and turn.speaker not in debaters:
            continue

        if len(turn.text.split()) < MIN_WORDS:
            continue

        # Clean text before splitting into sentences
        cleaned_text = _clean_text_for_chains(turn.text)
        sentences = _SENTENCE_SPLIT.split(cleaned_text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.split()) >= 3]

        if len(sentences) < 2:
            turn.argument_depth = 0.0
            continue

        # Embed all sentences for similarity checks
        sent_embs = [embedder.embed(s) for s in sentences]
        # Extract content words for shared-word check
        sent_content = [_get_content_words(s) for s in sentences]

        max_chain = 0.0
        current_chain = 0
        chain_has_conclusion = False

        for i, sent in enumerate(sentences):
            if i == 0:
                if _has_connector(sent):
                    current_chain = 1
                    chain_has_conclusion = _has_conclusion(sent)
                else:
                    current_chain = 0
                continue

            has_conn = _has_connector(sent)
            sim = float(np.dot(sent_embs[i], sent_embs[i - 1]))

            # Check shared content words (catches implicit links)
            shared_content = sent_content[i] & sent_content[i - 1]
            has_shared_words = len(shared_content) >= MIN_SHARED_CONTENT_WORDS

            # Chain continues if:
            # 1. Connector + similarity in Goldilocks zone, OR
            # 2. Shared content words + similarity above minimum (implicit link), OR
            # 3. Moderate similarity alone (implicit topical continuation in written text)
            continues = False
            if has_conn and CHAIN_SIM_MIN < sim < CHAIN_SIM_MAX:
                continues = True
            elif has_shared_words and sim > CHAIN_SIM_MIN:
                continues = True
            elif IMPLICIT_CONTINUATION_MIN < sim < CHAIN_SIM_MAX and current_chain > 0:
                continues = True  # can only extend existing chains, not start new ones

            if continues:
                if current_chain == 0:
                    current_chain = 2
                else:
                    current_chain += 1
                if _has_conclusion(sent):
                    chain_has_conclusion = True
            else:
                # Chain breaks — record with multiplier if concluded
                effective = current_chain * CONCLUSION_MULTIPLIER if chain_has_conclusion else current_chain
                max_chain = max(max_chain, effective)
                current_chain = 0
                chain_has_conclusion = False
                # This sentence might start a new chain
                if has_conn:
                    current_chain = 1
                    chain_has_conclusion = _has_conclusion(sent)

        # Final chain
        effective = current_chain * CONCLUSION_MULTIPLIER if chain_has_conclusion else current_chain
        max_chain = max(max_chain, effective)

        turn.argument_depth = round(min(max_chain / MAX_CHAIN, 1.0), 4)
