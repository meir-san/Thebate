from jinja2 import Environment, FileSystemLoader
from models import DebateResult
import os


def ms_to_timestamp(ms):
    """Convert milliseconds to MM:SS format."""
    total_seconds = int(ms) // 1000
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:02d}"


def ms_to_human(ms):
    """Convert milliseconds to human readable like '1h 42m'."""
    total_seconds = int(ms) // 1000
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    if hours > 0:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m"


def score_color(score):
    """Return CSS color based on score range."""
    if score >= 70:
        return "#22c55e"
    elif score >= 40:
        return "#eab308"
    return "#ef4444"


CLOSING_PHRASES = ["thank you", "thanks so much", "just before we end", "that's all", "to wrap up"]


def _group_flags(flags):
    """Group flags by (turn_index, flag_type). Returns list of dicts with flag, count, extras."""
    grouped = []
    seen = set()
    for flag in flags:
        key = (flag.turn_index, flag.flag_type)
        if key in seen:
            continue
        same = [f for f in flags if f.turn_index == flag.turn_index and f.flag_type == flag.flag_type]
        seen.add(key)
        grouped.append({
            "flag": same[0],
            "count": len(same),
            "extra": len(same) - 1,
        })
    return grouped


def _find_latest_flag(flags, turns_by_index, debaters):
    """Find the most recent flag from a debater whose turn doesn't start with a closing phrase."""
    for flag in reversed(flags):
        turn = turns_by_index[flag.turn_index]
        if debaters and turn.speaker not in debaters:
            continue
        text_lower = turn.text.lower().strip()
        if any(text_lower.startswith(phrase) for phrase in CLOSING_PHRASES):
            continue
        return flag, turn
    return None, None


def _build_env():
    env = Environment(loader=FileSystemLoader("templates"))
    env.filters["ms_to_timestamp"] = ms_to_timestamp
    env.filters["ms_to_human"] = ms_to_human
    env.filters["score_color"] = score_color
    return env


def render_report(result: DebateResult, output_path: str) -> None:
    env = _build_env()
    template = env.get_template("report.html.j2")

    all_flags = sorted(
        [f for t in result.turns for f in t.flags],
        key=lambda f: f.turn_index
    )
    turns_by_index = {t.index: t for t in result.turns}
    speakers = result.speakers

    # Pre-compute flag counts per speaker
    flag_counts = {}
    for t in result.turns:
        flag_counts[t.speaker] = flag_counts.get(t.speaker, 0) + len(t.flags)

    # Group flags by (turn_index, flag_type) for the flag log
    grouped_flags = _group_flags(all_flags)

    html = template.render(
        result=result,
        speakers=speakers,
        stats=result.stats,
        turns=result.turns,
        flags=all_flags,
        grouped_flags=grouped_flags,
        turns_by_index=turns_by_index,
        flag_counts=flag_counts,
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


def render_overlay(result: DebateResult, output_path: str) -> None:
    env = _build_env()
    template = env.get_template("overlay.html.j2")

    all_flags = sorted(
        [f for t in result.turns for f in t.flags],
        key=lambda f: f.turn_index
    )
    turns_by_index = {t.index: t for t in result.turns}
    speakers = result.speakers

    debaters = result.debaters
    latest_flag, latest_flag_turn = _find_latest_flag(all_flags, turns_by_index, debaters)

    html = template.render(
        result=result,
        speakers=speakers,
        debaters=debaters,
        stats=result.stats,
        turns=result.turns,
        flags=all_flags,
        turns_by_index=turns_by_index,
        latest_flag=latest_flag,
        latest_flag_turn=latest_flag_turn,
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
