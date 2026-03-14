"""Transcript preprocessor — transforms raw diarized turns into clean argument units.

Based on dialogue act classification research (Stolcke et al. 2000, Switchboard-DAMSL).

Steps:
  1. Classify each turn by dialogue act type (regex-based)
  2. Merge interrupted arguments (backchannel/fragment between same-speaker turns)
  3. Clean text within each turn (remove fillers, false starts, repeated words)
  4. Build argument exchanges (group consecutive turns into topic-linked exchanges)
  5. Mark turns for scoring exclusion (backchannels, fragments, merged turns)
"""
import re

import numpy as np
from models import Turn

# === Step 1: Dialogue act classification ===

# Backchannel words/phrases — turns consisting entirely of these
_BACKCHANNEL_PHRASES = {
    "yeah", "yes", "right", "okay", "ok", "uh huh", "uh-huh", "mm-hmm",
    "mm", "hmm", "sure", "yep", "yup", "exactly", "absolutely", "indeed",
    "oh", "ah", "i see", "got it", "go on", "fair enough",
}

# Claim verbs — if present in a short turn, it's not a backchannel
_CLAIM_VERBS = re.compile(
    r"\b(?:is|are|was|were|does|did|has|have|can|could|will|would|should|must|"
    r"proves?|shows?|demonstrates?|means?|causes?|makes?|creates?)\b",
    re.IGNORECASE,
)

_QUESTION_STARTS = re.compile(
    r"^(?:what|how|why|where|when|who|do you|don't you|is it|are you|"
    r"can you|would you|could you|don't you think|isn't it|aren't)\b",
    re.IGNORECASE,
)

_AGREEMENT_STARTS = re.compile(
    r"^(?:I agree|you(?:'re| are) right|that(?:'s| is) true|fair point|"
    r"good point|exactly right|you(?:'re| are) correct|that(?:'s| is) correct|"
    r"that(?:'s| is) a good point|that(?:'s| is) a fair point)\b",
    re.IGNORECASE,
)

_DISAGREEMENT_STARTS = re.compile(
    r"^(?:no[,. ]|that(?:'s| is) wrong|that(?:'s| is) not true|I disagree|"
    r"that(?:'s| is) incorrect|absolutely not|that(?:'s| is) false|"
    r"that(?:'s| is) not correct|wrong)\b",
    re.IGNORECASE,
)

# === Step 3: Text cleaning ===

_FILLER_WORDS = re.compile(
    r"\b(?:uh|um|uh+m*|like|you know|I mean|sort of|kind of|right|okay|ok|"
    r"so|well|basically|actually|literally|really|just|pretty much|"
    r"honestly|frankly|obviously|clearly|of course|anyway|anyways)\b[,.]?\s*",
    re.IGNORECASE,
)

_FILLER_START = re.compile(
    r"^(?:uh|um|like|so|well|I mean|you know|right|okay|ok)[,.]?\s+",
    re.IGNORECASE,
)

_REPEATED_WORDS = re.compile(r"\b(\w+)(?:\s+\1){2,}\b", re.IGNORECASE)

# False starts: repeated word sequences within proximity
_FALSE_STARTS = re.compile(r"\b(\w+(?:\s+\w+){0,3})\s+\1\b", re.IGNORECASE)


def classify_dialogue_act(turn: Turn) -> str:
    """Classify a turn's dialogue act type using regex patterns."""
    text = turn.text.strip()
    text_lower = text.lower().strip().rstrip(".!,")
    word_count = len(text.split())

    # BACKCHANNEL: entirely filler/agreement words, or short non-claim turns
    if text_lower in _BACKCHANNEL_PHRASES:
        return "backchannel"
    if word_count < 3 and not text.endswith("?") and not _CLAIM_VERBS.search(text):
        return "backchannel"

    # QUESTION: ends with ? or starts with question word
    if text.rstrip().endswith("?"):
        return "question"
    if _QUESTION_STARTS.match(text):
        return "question"

    # FRAGMENT: short non-question, non-backchannel
    if word_count < 5 and not text.endswith("?"):
        return "fragment"

    # AGREEMENT: starts with agreement marker + substantive content
    if _AGREEMENT_STARTS.match(text) and word_count > 10:
        return "agreement"

    # DISAGREEMENT: starts with disagreement marker + substantive content
    if _DISAGREEMENT_STARTS.match(text) and word_count > 10:
        return "disagreement"

    # STATEMENT: everything else
    return "statement"


def clean_turn_text(text: str) -> str:
    """Clean spoken language artifacts from turn text."""
    cleaned = text.strip()

    # Remove filler words at start (multiple passes)
    for _ in range(5):
        prev = cleaned
        m = _FILLER_START.match(cleaned)
        if m:
            cleaned = cleaned[m.end():]
        if cleaned == prev:
            break

    # Remove false starts ("they can't they can't distinguish" -> "they can't distinguish")
    cleaned = _FALSE_STARTS.sub(r"\1", cleaned)

    # Collapse repeated words ("no no no no" -> "no")
    cleaned = _REPEATED_WORDS.sub(r"\1", cleaned)

    # Strip trailing fragments (text after last sentence-ending punctuation
    # that is under 5 words and doesn't end with punctuation)
    sentences = re.split(r'(?<=[.!?])\s+', cleaned)
    if len(sentences) > 1:
        last = sentences[-1].strip()
        if not re.search(r'[.!?]$', last) and len(last.split()) < 5:
            cleaned = " ".join(sentences[:-1])

    # Collapse multiple spaces
    cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip()

    return cleaned if cleaned else text.strip()


def merge_interrupted_arguments(turns: list[Turn]) -> list[Turn]:
    """Merge turns interrupted by backchannels/fragments.

    When Speaker A's statement is interrupted by Speaker B's backchannel/fragment
    (<8 words), and Speaker A continues, merge Speaker A's turns.

    Applied recursively — 3 turns by A separated by 2 backchannels merge into 1.
    """
    if len(turns) < 3:
        return turns

    merged_indices: set[int] = set()  # indices of turns that were consumed by merging
    interrupt_indices: set[int] = set()  # indices of backchannel/fragment interrupts

    i = 0
    while i < len(turns) - 2:
        turn_a = turns[i]

        # Skip if already consumed
        if i in merged_indices or i in interrupt_indices:
            i += 1
            continue

        # Turn A must be substantive (statement or disagreement, >10 words)
        if turn_a.dialogue_act not in ("statement", "disagreement"):
            i += 1
            continue
        if len(turn_a.text.split()) <= 10:
            i += 1
            continue

        # Look ahead for interrupt + continuation pattern
        j = i + 1
        while j < len(turns) - 1:
            turn_b = turns[j]
            turn_a2 = turns[j + 1]

            # Turn B must be by different speaker, and short (<8 words)
            if turn_b.speaker == turn_a.speaker:
                break
            if len(turn_b.text.split()) >= 8:
                break
            if turn_b.dialogue_act not in ("backchannel", "fragment"):
                break

            # Turn A2 must be by the same speaker as Turn A
            if turn_a2.speaker != turn_a.speaker:
                break

            # Merge: concatenate text, extend timestamps
            turn_a.text = turn_a.text.rstrip() + " " + turn_a2.text.lstrip()
            turn_a.end_ms = turn_a2.end_ms

            # Mark the interrupt and the continuation as consumed
            turn_b.merged_into = turn_a.index
            turn_b.score_this = False
            interrupt_indices.add(j)

            turn_a2.merged_into = turn_a.index
            turn_a2.score_this = False
            merged_indices.add(j + 1)

            j += 2  # skip past the merged pair, look for more

        i += 1

    return turns


def build_exchanges(
    turns: list[Turn],
    turn_embeddings: dict[int, np.ndarray],
    debaters: list[str] | None = None,
    max_exchange_gap: int = 3,
    min_topic_similarity: float = 0.2,
) -> list[dict]:
    """Group consecutive turns into argument exchanges.

    An exchange starts when a speaker makes a STATEMENT or asks a QUESTION.
    It includes the opponent's response and follow-up turns until topic changes.
    """
    exchanges: list[dict] = []
    current_exchange: dict | None = None
    turns_since_topic = 0

    scorable_turns = [t for t in turns if t.score_this]

    for turn in scorable_turns:
        if debaters and turn.speaker not in debaters:
            continue

        if turn.index not in turn_embeddings:
            continue

        # Should we start a new exchange?
        start_new = False

        if current_exchange is None:
            start_new = True
        elif turns_since_topic >= max_exchange_gap:
            start_new = True
        else:
            # Check topic similarity to exchange opener
            topic_emb = current_exchange["topic_embedding"]
            turn_emb = turn_embeddings[turn.index]
            sim = float(np.dot(topic_emb, turn_emb))
            if sim < min_topic_similarity:
                start_new = True

        if start_new and turn.dialogue_act in ("statement", "question", "disagreement"):
            # Close current exchange
            if current_exchange is not None:
                exchanges.append(current_exchange)

            current_exchange = {
                "exchange_id": len(exchanges),
                "initiator": turn.speaker,
                "topic_embedding": turn_embeddings[turn.index],
                "turn_indices": [turn.index],
            }
            turn.exchange_id = len(exchanges)
            turns_since_topic = 0
        elif current_exchange is not None:
            current_exchange["turn_indices"].append(turn.index)
            turn.exchange_id = current_exchange["exchange_id"]
            turns_since_topic += 1

    # Close final exchange
    if current_exchange is not None:
        exchanges.append(current_exchange)

    # Build serializable exchange list (without numpy arrays)
    exchange_list = []
    for ex in exchanges:
        exchange_list.append({
            "exchange_id": ex["exchange_id"],
            "initiator": ex["initiator"],
            "turn_indices": ex["turn_indices"],
        })

    return exchange_list


def preprocess(
    turns: list[Turn],
    turn_embeddings: dict[int, np.ndarray] | None = None,
    debaters: list[str] | None = None,
) -> list[dict]:
    """Run all preprocessing steps on turns. Mutates turns in place.

    Returns list of exchange dicts for the output JSON.
    """
    # Step 1: Classify dialogue acts
    for turn in turns:
        turn.dialogue_act = classify_dialogue_act(turn)

    # Count dialogue acts before merging
    act_counts = {}
    for turn in turns:
        act_counts[turn.dialogue_act] = act_counts.get(turn.dialogue_act, 0) + 1
    print(f"  Dialogue acts: {act_counts}")

    # Step 2: Merge interrupted arguments
    merge_interrupted_arguments(turns)
    merged_count = sum(1 for t in turns if t.merged_into is not None)
    print(f"  Merged {merged_count} turns (backchannels/fragments absorbed into adjacent statements)")

    # Step 3: Clean text
    for turn in turns:
        turn.clean_text = clean_turn_text(turn.text)

    # Step 5: Mark turns for scoring exclusion
    for turn in turns:
        if turn.merged_into is not None:
            turn.score_this = False
        elif turn.dialogue_act == "backchannel":
            turn.score_this = False
        elif turn.dialogue_act == "fragment":
            turn.score_this = False
        else:
            turn.score_this = True

    excluded = sum(1 for t in turns if not t.score_this)
    scorable = sum(1 for t in turns if t.score_this)
    print(f"  Excluded {excluded} turns from scoring ({scorable} scorable)")

    # Step 4: Build exchanges (needs embeddings)
    exchanges = []
    if turn_embeddings:
        exchanges = build_exchanges(turns, turn_embeddings, debaters=debaters)
        print(f"  Built {len(exchanges)} argument exchanges")

    return exchanges
