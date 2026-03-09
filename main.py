import argparse
import os
import sys
from argparse import Namespace

from dotenv import load_dotenv


def parse_args():
    parser = argparse.ArgumentParser(
        description="DebateStats — Score the process of argument, not the content.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python main.py --url "https://youtube.com/watch?v=XXXX" --topic "The topic"
  python main.py --url "..." --topic "..." --speakers "A,B" --output-dir ./out/
  python main.py --url "..." --topic "..." --skip-ingest --output-dir ./out/
""",
    )
    parser.add_argument("--url", required=True, help="Full YouTube URL")
    parser.add_argument("--topic", required=True, help="Debate topic in plain English")
    parser.add_argument("--speakers", default=None, help="Comma-separated real names in order of first appearance")
    parser.add_argument("--debaters", default=None, help="Comma-separated names of speakers to score (subset of --speakers)")
    parser.add_argument("--output-dir", default="./output/", help="Directory for all output files (default: ./output/)")
    parser.add_argument("--adapter", default="assemblyai", choices=["assemblyai"], help="Transcription adapter (default: assemblyai)")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip Phase 1, use existing turns.json in output-dir")
    parser.add_argument("--skip-score", action="store_true", help="Skip Phase 2, use existing scored.json in output-dir")
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    turns_path = os.path.join(output_dir, "turns.json")
    scored_path = os.path.join(output_dir, "scored.json")
    report_path = os.path.join(output_dir, "report.html")
    overlay_path = os.path.join(output_dir, "overlay.html")

    # Phase 1: Ingestion
    if not args.skip_ingest:
        print("\n━━━ Phase 1: Ingestion ━━━")
        from phase1_ingest import run as run_ingest
        ingest_args = Namespace(
            url=args.url,
            topic=args.topic,
            speakers=args.speakers,
            debaters=args.debaters,
            output=turns_path,
            adapter=args.adapter,
        )
        run_ingest(ingest_args)
    else:
        # Only need turns.json if we're actually running Phase 2
        if not args.skip_score and not os.path.exists(turns_path):
            print(f"Error: --skip-ingest but {turns_path} not found")
            sys.exit(1)
        print(f"\n━━━ Phase 1: Skipped ━━━")

    # Phase 2: Scoring
    if not args.skip_score:
        print("\n━━━ Phase 2: Scoring ━━━")
        from phase2_score import run as run_score
        score_args = Namespace(
            input=turns_path,
            output=scored_path,
            threshold_engagement=None,
            threshold_dodge=None,
            threshold_drift=None,
        )
        run_score(score_args)
    else:
        if not os.path.exists(scored_path):
            print(f"Error: --skip-score but {scored_path} not found")
            sys.exit(1)
        print(f"\n━━━ Phase 2: Skipped (using {scored_path}) ━━━")

    # Phase 3: Rendering
    print("\n━━━ Phase 3: Rendering ━━━")
    from phase3_render import run as run_render
    render_args = Namespace(
        input=scored_path,
        report=report_path,
        overlay=overlay_path,
    )
    run_render(render_args)

    # Final summary
    print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✓ Debate analysis complete
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Report:   {report_path}
  Overlay:  {overlay_path}  (OBS Browser Source, 1920×1080)
  Raw data: {scored_path}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━""")


if __name__ == "__main__":
    main()
