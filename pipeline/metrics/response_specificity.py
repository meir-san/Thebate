import re

from models import Turn

# Named entities: 2+ consecutive capitalized words, not at sentence start
_RE_NAMED_ENTITY = re.compile(r'(?<![.!?]\s)(?<!\A)(?<!\n)\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b')

# Numbers, dates, measurements, percentages
_RE_NUMBER = re.compile(r'\b\d[\d,]*(?:\.\d+)?(?:\s*(?:%|percent|million|billion|thousand|kg|km|miles|degrees|years))?\b')

# Technical/domain terms: >8 chars with domain-specific suffixes
_RE_TECHNICAL = re.compile(
    r'\b[a-zA-Z]{4,}(?:tion|ment|ical|ology|phere|metric|istic|ization|alism)\b',
    re.IGNORECASE,
)

# Specificity phrases
_RE_SPECIFICITY = re.compile(
    r'\b(?:specifically|exactly|precisely|in particular|measured at|'
    r'calculated to|to be exact|in the case of)\b',
    re.IGNORECASE,
)

_WORD_RE = re.compile(r"[a-zA-Z']+")

# Top ~200 most common content words — any content word NOT in this set
# and longer than 6 chars counts as rare/specific
COMMON_WORDS = {
    "people", "time", "thing", "things", "work", "world", "make", "good",
    "know", "take", "come", "want", "look", "give", "tell", "call", "keep",
    "point", "fact", "part", "place", "case", "week", "group", "number",
    "area", "course", "problem", "hand", "help", "change", "large", "small",
    "different", "still", "back", "great", "right", "wrong", "real", "long",
    "little", "same", "country", "school", "state", "family", "student",
    "system", "program", "question", "government", "company", "story",
    "side", "kind", "head", "house", "service", "friend", "father", "power",
    "hour", "game", "line", "city", "community", "name", "president",
    "team", "minute", "idea", "body", "information", "level", "able",
    "force", "money", "children", "development", "night", "human", "water",
    "process", "room", "mother", "market", "study", "life", "book", "child",
    "form", "door", "experience", "teacher", "result", "office", "woman",
    "report", "decision", "situation", "role", "person", "girl", "road",
    "food", "table", "face", "nature", "building", "action", "value",
    "issue", "party", "special", "manager", "project", "field", "million",
    "support", "record", "morning", "language", "interest", "class", "reason",
    "order", "anything", "position", "member", "meeting", "important",
    "someone", "nothing", "believe", "actually", "simple", "money",
    "history", "picture", "today", "sense", "example", "computer",
    "business", "possible", "personal", "national", "social", "really",
    "everything", "always", "something", "around", "think", "every",
    "again", "never", "start", "going", "other", "might", "about",
    "after", "being", "could", "should", "would", "where", "their",
    "there", "these", "those", "which", "while", "before", "between",
    "through", "under", "years", "first", "second", "third", "last",
    "because", "since", "until", "already", "enough", "together",
    "however", "another", "whether", "though", "without", "during",
    "certain", "general", "public", "local", "likely", "simply",
    "perhaps", "rather", "recent", "several", "either", "current",
    "often", "moment", "former", "similar", "within", "political",
    "matter", "center", "policy", "research", "education", "practice",
    "consider", "continue", "follow", "provide", "include", "require",
    "suggest", "produce", "receive", "control", "protect", "accept",
    "present", "common", "myself",
}

MAX_SPECIFICITY = 3.0  # normalize by this


def score_response_specificity(
    turns: list[Turn],
    debaters: list[str] | None = None,
) -> None:
    """Measure information density of responses including lexical specificity."""
    for i, turn in enumerate(turns):
        if debaters and turn.speaker not in debaters:
            continue

        # Only score turns that follow an opponent
        has_opponent_before = False
        for j in range(i - 1, -1, -1):
            if turns[j].speaker != turn.speaker:
                has_opponent_before = True
                break
        if not has_opponent_before:
            continue

        word_count = len(turn.text.split())
        if word_count < 10:
            continue

        named_entity_count = len(_RE_NAMED_ENTITY.findall(turn.text))
        number_count = len(_RE_NUMBER.findall(turn.text))
        technical_term_count = len(_RE_TECHNICAL.findall(turn.text))
        specificity_phrase_count = len(_RE_SPECIFICITY.findall(turn.text))

        # Count rare/specific words: content words >6 chars not in common set
        words = _WORD_RE.findall(turn.text.lower())
        content_words = [w for w in words if len(w) > 4]
        rare_word_count = sum(
            1 for w in content_words
            if len(w) > 6 and w not in COMMON_WORDS
        )

        raw = (
            named_entity_count * 2
            + number_count * 2
            + technical_term_count
            + specificity_phrase_count
            + rare_word_count * 1.5
        ) / max(word_count / 10, 1)

        turn.response_specificity = round(min(raw / MAX_SPECIFICITY, 1.0), 4)
