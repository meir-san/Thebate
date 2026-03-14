import re

import numpy as np
from models import Turn, SpeakerStats

_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')

_CONCLUSION_INDICATORS = re.compile(
    r"\b(?:therefore|thus|hence|consequently|it follows that|"
    r"which means|this proves|this shows|this demonstrates|"
    r"that's why|which is why|this means|meaning that|"
    r"which distinguishes|which confirms|which explains|"
    r"indicating that|confirming that|in other words|"
    r"this is why|this confirms)\b",
    re.IGNORECASE,
)

_PREMISE_INDICATORS = re.compile(
    r"\b(?:because|since|given that|due to|based on|"
    r"for example|for instance|according to|as shown by|"
    r"as evidenced by|research shows|studies show|evidence shows|"
    r"the data shows)\b",
    re.IGNORECASE,
)

DEDUP_THRESHOLD = 0.85       # sentences above this are near-duplicates
SIMILARITY_EDGE_THRESHOLD = 0.5  # topic edges on deduplicated sentences
SUPPORT_EDGE_THRESHOLD = 0.4
MAX_SENTENCES_PER_SPEAKER = 500


def _split_sentences(text: str) -> list[str]:
    sentences = _SENTENCE_SPLIT.split(text.strip())
    return [s.strip() for s in sentences if s.strip() and len(s.split()) >= 3]


def _deduplicate_sentences(
    sentences: list[str],
    emb_matrix: np.ndarray,
) -> tuple[list[str], np.ndarray, float]:
    """Cluster sentences by >0.85 similarity. Return representatives + unique ratio.

    Returns (deduped_sentences, deduped_embeddings, unique_claims_ratio).
    """
    n = len(sentences)
    if n == 0:
        return [], np.array([]), 0.0

    sim_matrix = emb_matrix @ emb_matrix.T

    # Greedy clustering: assign each sentence to first cluster with sim > threshold
    cluster_assignments = [-1] * n
    clusters: list[list[int]] = []

    for i in range(n):
        assigned = False
        for ci, members in enumerate(clusters):
            rep = members[0]  # compare to representative
            if sim_matrix[i, rep] > DEDUP_THRESHOLD:
                clusters[ci].append(i)
                cluster_assignments[i] = ci
                assigned = True
                break
        if not assigned:
            cluster_assignments[i] = len(clusters)
            clusters.append([i])

    # Pick longest sentence as representative for each cluster
    deduped_sentences: list[str] = []
    deduped_indices: list[int] = []
    for members in clusters:
        best_idx = max(members, key=lambda idx: len(sentences[idx]))
        deduped_sentences.append(sentences[best_idx])
        deduped_indices.append(best_idx)

    deduped_embs = emb_matrix[deduped_indices]
    unique_ratio = len(clusters) / n if n > 0 else 0.0

    return deduped_sentences, deduped_embs, unique_ratio


def _find_connected_components(n: int, edges: list[tuple[int, int]]) -> list[set[int]]:
    """Simple union-find for connected components."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b in edges:
        union(a, b)

    components: dict[int, set[int]] = {}
    for i in range(n):
        root = find(i)
        components.setdefault(root, set()).add(i)
    return list(components.values())


def _longest_support_path(n: int, support_adj: dict[int, list[int]]) -> int:
    """DFS for longest path in the support DAG."""
    if not support_adj:
        return 0
    memo: dict[int, int] = {}

    def dfs(node: int, visited: set[int]) -> int:
        if node in memo:
            return memo[node]
        best = 0
        for nxt in support_adj.get(node, []):
            if nxt not in visited:
                visited.add(nxt)
                best = max(best, 1 + dfs(nxt, visited))
                visited.discard(nxt)
        memo[node] = best
        return best

    longest = 0
    for start in support_adj:
        longest = max(longest, dfs(start, {start}))
    return longest


def score_argument_graph(
    turns: list[Turn],
    embedder,
    debaters: list[str] | None = None,
) -> dict[str, float]:
    """Build per-speaker argument graph on deduplicated sentences and score coherence.

    Returns dict mapping speaker -> graph_coherence_score (for SpeakerStats).
    """
    speakers = debaters or list({t.speaker for t in turns})
    results: dict[str, float] = {}

    for speaker in speakers:
        speaker_turns = [t for t in turns if t.speaker == speaker]
        if not speaker_turns:
            results[speaker] = 0.0
            continue

        # Collect sentences across all turns
        all_sentences: list[str] = []
        for t in speaker_turns:
            all_sentences.extend(_split_sentences(t.text))

        if len(all_sentences) < 2:
            results[speaker] = 0.0
            continue

        # Sample evenly if too many sentences
        if len(all_sentences) > MAX_SENTENCES_PER_SPEAKER:
            indices = np.linspace(0, len(all_sentences) - 1,
                                  MAX_SENTENCES_PER_SPEAKER, dtype=int)
            all_sentences = [all_sentences[i] for i in indices]

        total_sentences = len(all_sentences)

        # Embed all sentences in batch
        raw_emb_matrix = np.array(embedder.embed_batch(all_sentences))

        # Deduplicate: cluster by >0.85 similarity, keep longest representative
        deduped_sentences, deduped_embs, unique_claims_ratio = _deduplicate_sentences(
            all_sentences, raw_emb_matrix,
        )

        n = len(deduped_sentences)
        if n < 2:
            # Almost everything is a repeat — score reflects low uniqueness
            results[speaker] = round(unique_claims_ratio * 0.4, 4)
            continue

        # Build graph on deduplicated sentences only
        sim_matrix = deduped_embs @ deduped_embs.T

        topic_edges: list[tuple[int, int]] = []
        support_adj: dict[int, list[int]] = {}

        # Upper triangle for topic edges
        i_idx, j_idx = np.triu_indices(n, k=1)
        topic_mask = sim_matrix[i_idx, j_idx] > SIMILARITY_EDGE_THRESHOLD
        for idx in np.where(topic_mask)[0]:
            topic_edges.append((int(i_idx[idx]), int(j_idx[idx])))

        # Support edges: sentence j has conclusion/premise indicator and sim(i, j) > threshold
        has_reasoning = [bool(_CONCLUSION_INDICATORS.search(s) or _PREMISE_INDICATORS.search(s))
                         for s in deduped_sentences]
        for j in range(n):
            if not has_reasoning[j]:
                continue
            for i in range(n):
                if i == j:
                    continue
                if sim_matrix[i, j] > SUPPORT_EDGE_THRESHOLD:
                    support_adj.setdefault(i, []).append(j)
                    if i < j:
                        topic_edges.append((i, j))
                    else:
                        topic_edges.append((j, i))

        topic_edges = list(set(topic_edges))

        # 1. Graph connectivity
        connected_nodes: set[int] = set()
        for a, b in topic_edges:
            connected_nodes.add(a)
            connected_nodes.add(b)
        graph_connectivity = len(connected_nodes) / n if n > 0 else 0.0

        # 2. Average cluster size
        components = _find_connected_components(n, topic_edges)
        multi_components = [c for c in components if len(c) > 1]
        avg_cluster_size = (
            sum(len(c) for c in multi_components) / len(multi_components)
            if multi_components else 0.0
        )

        # 3. Support chain depth
        support_chain_depth = _longest_support_path(n, support_adj)

        # 4. Topical breadth — how many distinct topic clusters does this speaker span?
        # Cluster deduplicated sentences at 0.4 similarity (broader than dedup threshold)
        BREADTH_CLUSTER_THRESHOLD = 0.4
        breadth_clusters: list[list[int]] = []
        for i_node in range(n):
            assigned = False
            for ci, members in enumerate(breadth_clusters):
                rep = members[0]
                if sim_matrix[i_node, rep] > BREADTH_CLUSTER_THRESHOLD:
                    breadth_clusters[ci].append(i_node)
                    assigned = True
                    break
            if not assigned:
                breadth_clusters.append([i_node])
        num_topic_clusters = len(breadth_clusters)
        topical_breadth = min(num_topic_clusters / 10, 1.0)

        # Combined score — uniqueness penalizes repetition, breadth rewards range
        graph_coherence_score = (
            graph_connectivity * 0.10
            + min(avg_cluster_size / 5, 1.0) * 0.10
            + min(support_chain_depth / 4, 1.0) * 0.25
            + unique_claims_ratio * 0.30
            + topical_breadth * 0.25
        )

        results[speaker] = round(graph_coherence_score, 4)

    return results
