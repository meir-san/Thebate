import argparse
import json
import sys

from models import DebateResult
from renderer import render_report, render_overlay


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 3: Render scored debate data to HTML report and overlay.")
    parser.add_argument("--input", default="scored.json", help="Input scored JSON from phase2 (default: scored.json)")
    parser.add_argument("--report", default="report.html", help="Output report HTML path (default: report.html)")
    parser.add_argument("--overlay", default="overlay.html", help="Output overlay HTML path (default: overlay.html)")
    return parser.parse_args()


def main():
    args = parse_args()

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

    if not result.stats:
        print("Error: input has no stats. Run phase2_score.py first.")
        sys.exit(1)

    render_report(result, args.report)
    render_overlay(result, args.overlay)

    print(f"✓ Report saved to {args.report}")
    print(f"✓ Overlay saved to {args.overlay} (1920×1080, use as OBS Browser Source)")


if __name__ == "__main__":
    main()
