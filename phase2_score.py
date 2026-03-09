import argparse
import json
import sys

from dotenv import load_dotenv

import config
from models import DebateResult
from pipeline.metrics.dodge import extract_questions


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2: Score debate turns on argumentation metrics.")
    parser.add_argument("--input", default="turns.json", help="Input JSON from phase1 (default: turns.json)")
    parser.add_argument("--output", default="scored.json", help="Output scored JSON (default: scored.json)")
    parser.add_argument("--threshold-engagement", type=float, default=None, help="Override engagement threshold")
    parser.add_argument("--threshold-dodge", type=float, default=None, help="Override dodge threshold")
    parser.add_argument("--threshold-drift", type=float, default=None, help="Override topic drift threshold")
    return parser.parse_args()


def run(args):
    """Core scoring logic. Called by main() or main.py wrapper."""
    load_dotenv()

    # Override config thresholds if provided
    if args.threshold_engagement is not None:
        config.THRESHOLD_ENGAGEMENT = args.threshold_engagement
    if args.threshold_dodge is not None:
        config.THRESHOLD_DODGE = args.threshold_dodge
    if args.threshold_drift is not None:
        config.THRESHOLD_TOPIC_DRIFT = args.threshold_drift

    # Load input
    try:
        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {args.input} not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON: {e}")
        sys.exit(1)

    result = DebateResult.from_dict(data)
    print(f"Loaded {len(result.turns)} turns for speakers: {result.speakers}")

    # Initialize embedder
    try:
        from pipeline.embedder import Embedder
        embedder = Embedder(config.EMBEDDING_MODEL)
    except Exception as e:
        print(f"Could not download embedding model. If offline, run once with internet to cache the model.")
        print(f"Details: {e}")
        sys.exit(1)

    # Batch 1: embed all turn texts
    print("Embedding turn texts...")
    turn_texts = [t.text for t in result.turns]
    turn_embs_array = embedder.embed_batch(turn_texts)
    turn_embeddings: dict[int, any] = {
        t.index: turn_embs_array[i] for i, t in enumerate(result.turns)
    }

    # Batch 2: embed the debate topic
    print("Embedding debate topic...")
    topic_embedding = embedder.embed(result.topic)

    # Batch 3: extract and embed all questions
    print("Extracting and embedding questions...")
    question_strings: list[tuple[str, str]] = []  # (key, text)
    for turn in result.turns:
        questions = extract_questions(turn.text)
        for qi, q in enumerate(questions):
            question_strings.append((f"{turn.index}_{qi}", q))

    if question_strings:
        q_texts = [qs[1] for qs in question_strings]
        q_embs_array = embedder.embed_batch(q_texts)
        question_embeddings: dict[str, any] = {
            question_strings[i][0]: q_embs_array[i]
            for i in range(len(question_strings))
        }
    else:
        question_embeddings = {}

    print(f"Embedded {len(result.turns)} turns, 1 topic, {len(question_strings)} questions\n")

    # Run metrics
    from pipeline.metrics.claim_ratio import score_claims
    from pipeline.metrics.engagement import score_engagement
    from pipeline.metrics.dodge import score_dodges
    from pipeline.metrics.topic_drift import score_topic_drift
    from scorer import score_debate

    print("Scoring claims...")
    score_claims(result.turns)

    print("Scoring engagement...")
    score_engagement(result.turns, turn_embeddings, debaters=result.debaters)

    print("Scoring dodges...")
    score_dodges(result.turns, turn_embeddings, question_embeddings, debaters=result.debaters)

    print("Scoring topic drift...")
    score_topic_drift(result.turns, turn_embeddings, topic_embedding)

    print("Building speaker stats...")
    score_debate(result)

    # Check for speakers with zero eligible turns
    for speaker, stats in result.stats.items():
        if stats.turn_count == 0:
            print(f"Warning: {speaker} has zero turns — stats may be unreliable")

    # Save output
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    # Print summary table
    print(f"\n{'Speaker':<20} {'Score':>6} {'Eng':>6} {'Dodges':>12} {'Claims':>14} {'Drift':>6}")
    for speaker, stats in result.stats.items():
        dodge_str = f"{stats.total_dodges}/{stats.questions_faced}"
        dodge_pct = f"({stats.dodge_rate:.0%})" if stats.questions_faced > 0 else "(n/a)"
        claim_str = f"{stats.supported_claims}/{stats.total_claims}"
        claim_pct = f"({stats.claim_support_ratio:.0%})" if stats.total_claims > 0 else "(n/a)"
        print(
            f"{speaker:<20} {stats.overall_score:>6.1f} "
            f"{stats.avg_engagement:>6.2f} "
            f"{dodge_str:>6} {dodge_pct:<5} "
            f"{claim_str:>6} {claim_pct:<5}  "
            f"{stats.avg_topic_drift:>5.2f}"
        )

    print(f"\nSaved to: {args.output}")
    print(f"\n→ Next: python phase3_render.py --input {args.output}")


def main():
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
