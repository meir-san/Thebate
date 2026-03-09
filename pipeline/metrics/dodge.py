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
    """
    threshold = config.THRESHOLD_DODGE

    for i, turn in enumerate(turns):
        # Only count questions from debaters
        if debaters and turn.speaker not in debaters:
            continue

        questions = extract_questions(turn.text)
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
