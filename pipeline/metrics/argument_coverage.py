import numpy as np
from models import Turn

CLUSTER_THRESHOLD = 0.55  # similarity to merge into existing cluster
COVERAGE_THRESHOLD = 0.3  # similarity to consider an argument "covered"
MIN_WORDS = 15


def score_argument_coverage(
    turns: list[Turn],
    turn_embeddings: dict[int, np.ndarray],
    debaters: list[str] | None = None,
) -> dict[str, float]:
    """Measure what proportion of opponent's distinct arguments a speaker addresses.

    Returns dict mapping speaker -> argument_coverage_score (for SpeakerStats).
    """
    speakers = debaters or list({t.speaker for t in turns})
    results: dict[str, float] = {}

    for speaker in speakers:
        # Collect opponent turns with sufficient length
        opponent_turns = [
            t for t in turns
            if t.speaker != speaker
            and (not debaters or t.speaker in debaters)
            and len(t.text.split()) >= MIN_WORDS
            and t.index in turn_embeddings
        ]

        if not opponent_turns:
            results[speaker] = 0.0
            continue

        # Cluster opponent turns by similarity (greedy)
        clusters: list[list[int]] = []  # list of lists of turn indices
        centroids: list[np.ndarray] = []

        for t in opponent_turns:
            emb = turn_embeddings[t.index]
            merged = False
            for ci, centroid in enumerate(centroids):
                sim = float(np.dot(emb, centroid))
                if sim > CLUSTER_THRESHOLD:
                    clusters[ci].append(t.index)
                    # Update centroid as running mean
                    n = len(clusters[ci])
                    new_centroid = centroid * ((n - 1) / n) + emb * (1 / n)
                    norm = np.linalg.norm(new_centroid)
                    if norm > 0:
                        new_centroid /= norm
                    centroids[ci] = new_centroid
                    merged = True
                    break
            if not merged:
                clusters.append([t.index])
                centroids.append(emb.copy())

        total_clusters = len(clusters)
        if total_clusters == 0:
            results[speaker] = 0.0
            continue

        # Check coverage: does any of this speaker's turns address each cluster?
        speaker_turns = [
            t for t in turns
            if t.speaker == speaker
            and t.index in turn_embeddings
        ]

        if not speaker_turns:
            results[speaker] = 0.0
            continue

        # Build speaker embedding matrix for vectorized comparison
        speaker_embs = np.array([turn_embeddings[t.index] for t in speaker_turns])

        covered = 0
        for centroid in centroids:
            sims = speaker_embs @ centroid
            if np.max(sims) > COVERAGE_THRESHOLD:
                covered += 1

        results[speaker] = round(covered / total_clusters, 4)

    return results
