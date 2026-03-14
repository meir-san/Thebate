import re

import numpy as np
from models import Turn, Flag
import config

QUESTION_STARTERS = [
    "do you", "did you", "what ", "why ", "how ", "would you",
    "can you", "is it ", "are you", "don't you", "doesn't that",
    "isn't it", "weren't you", "won't you", "will you", "have you",
    "could you", "should you", "does that", "do they", "is there",
]

FILLER_QUESTIONS = [
    r"^(but )?what's your thoughts?",
    r"^what do you think\??$",
    r"^and guess what\??$",
    r"^excuse me\??$",
    r"^you hear me\??$",
    r"^do you know what i (do|mean)\??$",
    r"^(so )?what about that\??$",
    r"^(ok|okay|right|so|well)\??$",
    r"^now you (might|may|could) say",
    r"^are you (saying|telling me|serious)",
    r"^(so )?what were they supposed to do\??$",
    r"^remember (the|that|when)",
]

_FILLER_RES = [re.compile(p, re.IGNORECASE) for p in FILLER_QUESTIONS]

STOPWORDS = {
    "about", "after", "also", "been", "before", "being", "between", "both",
    "could", "does", "doing", "down", "during", "each", "even", "every",
    "from", "have", "having", "here", "into", "just", "know", "like",
    "make", "many", "more", "most", "much", "must", "only", "other",
    "over", "same", "should", "some", "such", "than", "that", "their",
    "them", "then", "there", "these", "they", "thing", "things", "think",
    "this", "those", "through", "very", "want", "were", "what", "when",
    "where", "which", "while", "will", "with", "would", "your", "you're",
    "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't",
    "won't", "wouldn't", "can't", "couldn't", "shouldn't",
}


def _extract_key_terms(text: str) -> set[str]:
    """Extract content words (>4 chars, not stopwords) from text."""
    words = re.findall(r"[a-zA-Z']+", text.lower())
    return {w for w in words if len(w) > 4 and w not in STOPWORDS}


def _has_keyword_overlap(question: str, response: str, min_overlap: int = 2) -> bool:
    """Return True if the response contains enough key terms from the question."""
    q_terms = _extract_key_terms(question)
    if len(q_terms) < min_overlap:
        return False
    r_terms = _extract_key_terms(response)
    overlap = q_terms & r_terms
    return len(overlap) >= min_overlap


def _is_filler_question(q: str) -> bool:
    """Return True if the question is too short or matches a filler pattern."""
    if len(q.split()) < 6:
        return True
    return any(r.search(q) for r in _FILLER_RES)


def extract_questions(text: str) -> list[str]:
    """
    Returns list of question strings from a turn's text.
    Two detection methods:
    1. Sentence ends with '?'
    2. Sentence starts with a question starter phrase (for transcripts lacking punctuation)
    """
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    questions = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if s.endswith('?'):
            questions.append(s)
        elif any(s.lower().startswith(starter) for starter in QUESTION_STARTERS):
            questions.append(s)
    return questions


def score_dodges(
    turns: list[Turn],
    turn_embeddings: dict[int, np.ndarray],
    question_embeddings: dict[str, np.ndarray],
    debaters: list[str] | None = None,
) -> None:
    """Mutates turns in place. Sets turn.is_dodge and appends to turn.flags.
    If debaters is provided, only extracts questions from turns where the questioner
    is in debaters. Moderator questions are not scored as dodges.

    If speech_act data exists, also counts opponent "challenge" turns where
    the speaker's response has responds_to_opponent == False as dodges.
    """
    # Structure-based dodge detection (supplements regex)
    has_structure = any(t.speech_act is not None for t in turns)
    if has_structure:
        for i, turn in enumerate(turns):
            if debaters and turn.speaker not in debaters:
                continue
            if turn.speech_act != "challenge":
                continue

            # Find the next turn by a different speaker (the responder)
            responder_turn = None
            for j in range(i + 1, len(turns)):
                if turns[j].speaker != turn.speaker:
                    responder_turn = turns[j]
                    break
            if responder_turn is None:
                continue
            if debaters and responder_turn.speaker not in debaters:
                continue

            if responder_turn.responds_to_opponent is False:
                responder_turn.is_dodge = True
                responder_turn.flags.append(Flag(
                    turn_index=responder_turn.index,
                    flag_type="dodge",
                    score=0.0,
                    threshold=0.0,
                    explanation=(
                        f"Challenge from {turn.speaker} not addressed — "
                        f"responds_to_opponent=False"
                    ),
                ))

    threshold = config.THRESHOLD_DODGE

    for i, turn in enumerate(turns):
        # Only count questions from debaters
        if debaters and turn.speaker not in debaters:
            continue

        questions = extract_questions(turn.text)
        if not questions:
            continue

        # Rhetorical detection: 4+ questions in one turn = all rhetorical, skip
        if len(questions) > 3:
            continue

        # Filter out filler questions
        questions = [q for q in questions if not _is_filler_question(q)]
        if not questions:
            continue

        # Find the next turn by a different speaker
        responder_turn = None
        for j in range(i + 1, len(turns)):
            if turns[j].speaker != turn.speaker:
                responder_turn = turns[j]
                break

        if responder_turn is None:
            continue

        # Skip if responder's turn is too short to meaningfully analyze
        if len(responder_turn.text.split()) < config.MIN_WORDS_ENGAGEMENT:
            continue

        responder_emb = turn_embeddings[responder_turn.index]

        for qi, question_text in enumerate(questions):
            key = f"{turn.index}_{qi}"
            if key not in question_embeddings:
                continue
            q_emb = question_embeddings[key]
            similarity = float(np.dot(q_emb, responder_emb))

            if similarity < threshold:
                # Skip if response shares key terms with the question
                if _has_keyword_overlap(question_text, responder_turn.text):
                    continue
                responder_turn.is_dodge = True
                q_preview = question_text[:80] + "..." if len(question_text) > 80 else question_text
                responder_turn.flags.append(Flag(
                    turn_index=responder_turn.index,
                    flag_type="dodge",
                    score=similarity,
                    threshold=threshold,
                    explanation=(
                        f"Question: \"{q_preview}\" — "
                        f"Response similarity: {similarity:.2f} (threshold: {threshold:.2f})"
                    )
                ))
