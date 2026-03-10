"""Shared interactive speaker assignment logic.

Used by both assign_speakers.py (standalone) and phase1_ingest.py (after transcription).
"""


def _get_samples(turns, speaker_label: str, n: int = 3) -> list[dict]:
    """Pick n sample turns from beginning, middle, and end of a speaker's turns."""
    speaker_turns = [t for t in turns if t.speaker == speaker_label]
    if not speaker_turns:
        return []
    if len(speaker_turns) <= n:
        return speaker_turns
    step = max(1, (len(speaker_turns) - 1) // (n - 1))
    indices = [0]
    for i in range(1, n - 1):
        indices.append(i * step)
    indices.append(len(speaker_turns) - 1)
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            unique.append(idx)
    return [speaker_turns[i] for i in unique[:n]]


def _preview(turn) -> str:
    """First 30 words of a turn's text."""
    words = turn.text.split()[:30]
    preview = " ".join(words)
    if len(turn.text.split()) > 30:
        preview += "..."
    return preview


def assign_speakers_interactive(turns) -> dict[str, str]:
    """Interactive speaker assignment. Shows samples and prompts for names.

    Args:
        turns: list of Turn objects (or dicts with .speaker and .text)

    Returns:
        mapping dict: {old_label: new_name} for speakers that were renamed.
        Speakers where user typed 'skip' are not in the mapping.
    """
    # Find unique speaker labels in order of first appearance
    labels = []
    for t in turns:
        if t.speaker not in labels:
            labels.append(t.speaker)

    print(f"\n{'='*60}")
    print(f"Speaker Assignment — {len(labels)} speakers detected")
    print(f"{'='*60}")

    mapping = {}

    for label in labels:
        samples = _get_samples(turns, label)
        turn_count = sum(1 for t in turns if t.speaker == label)

        print(f"\n  {label} ({turn_count} turns)")
        print(f"  {'─'*50}")
        positions = ["beginning", "middle", "end"]
        for i, sample in enumerate(samples):
            pos = positions[i] if i < len(positions) else f"sample {i+1}"
            print(f"    [{pos}] \"{_preview(sample)}\"")

        name = input(f"  Name for {label} (or 'skip'): ").strip()
        if name and name.lower() != "skip":
            mapping[label] = name
        else:
            print(f"  → Keeping as {label}")

    return mapping


def apply_mapping(turns, mapping: dict[str, str]) -> None:
    """Apply speaker name mapping to turns in place."""
    for t in turns:
        if t.speaker in mapping:
            t.speaker = mapping[t.speaker]
