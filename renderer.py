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

    html = template.render(
        result=result,
        speakers=speakers,
        stats=result.stats,
        turns=result.turns,
        flags=all_flags,
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

    latest_flag = all_flags[-1] if all_flags else None
    latest_flag_turn = turns_by_index[latest_flag.turn_index] if latest_flag else None

    html = template.render(
        result=result,
        speakers=speakers,
        stats=result.stats,
        turns=result.turns,
        flags=all_flags,
        turns_by_index=turns_by_index,
        latest_flag=latest_flag,
        latest_flag_turn=latest_flag_turn,
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
