# Thresholds
THRESHOLD_ENGAGEMENT = 0.25
THRESHOLD_DODGE = 0.20        # Lower than engagement — real responses in conversational speech
                               # often score 0.25–0.35 naturally, so 0.20 avoids false positives
THRESHOLD_TOPIC_DRIFT = 0.78

# Short turn word cutoffs — turns below these are excluded from scoring
MIN_WORDS_ENGAGEMENT = 10     # Short acknowledgments have no semantic content to compare
MIN_WORDS_TOPIC_DRIFT = 15    # Short generic turns are semantically distant from any topic

# New metric thresholds
THRESHOLD_CORRECTION = 0.20
THRESHOLD_CONSISTENCY = 0.25
MIN_WORDS_EVIDENCE = 50          # Flag turns over this length with zero evidence markers

# Score weights — must sum to 100
# correction and consistency excluded from formula until reliable
SCORE_WEIGHTS = {
    "engagement": 25,
    "dodge": 20,
    "reasoning": 20,
    "drift": 15,
    "concession": 10,
    "evidence": 10,
}

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Reasoning connectors used by claim_ratio metric
REASONING_CONNECTORS = [
    "because", "therefore", "since", "thus", "hence",
    "as a result", "this means", "which means", "evidence shows",
    "research shows", "studies show", "for example", "for instance",
    "data shows", "according to", "this proves", "which proves",
    "this demonstrates", "which demonstrates", "the reason is",
    "that's why", "and so", "which is why"
]
