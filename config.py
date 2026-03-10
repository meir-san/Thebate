# Thresholds
THRESHOLD_ENGAGEMENT = 0.25
THRESHOLD_DODGE = 0.12        # Conservative — catches real dodges (topic changes) without flagging
                               # responses that address the question with different framing
THRESHOLD_TOPIC_DRIFT = 0.78

# Short turn word cutoffs — turns below these are excluded from scoring
MIN_WORDS_ENGAGEMENT = 10     # Short acknowledgments have no semantic content to compare
MIN_WORDS_TOPIC_DRIFT = 15    # Short generic turns are semantically distant from any topic

# New metric thresholds
THRESHOLD_CORRECTION = 0.20
THRESHOLD_CONSISTENCY = 0.25
MIN_WORDS_EVIDENCE = 50          # Flag turns over this length with zero evidence markers

# Score weights — must sum to 100
SCORE_WEIGHTS = {
    "premise_sufficiency": 15,
    "engagement_quality": 12,
    "correction": 12,
    "fallacy_free": 12,
    "argument_depth": 8,
    "response_specificity": 8,
    "opponent_engagement": 8,
    "evidence": 8,
    "dodge": 7,
    "scheme_diversity": 5,
    "drift": 3,
    "concession": 2,
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
