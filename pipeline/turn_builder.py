from models import Turn

# Maximum gap (ms) between consecutive segments by the same speaker
# before starting a new turn.
MAX_MERGE_GAP_MS = 2000

# Maximum duration (ms) for a single turn. If a merged turn exceeds this,
# force-split at the next segment boundary. Prevents 14-minute monologues
# from becoming single turns (important for per-turn metric scoring).
MAX_TURN_DURATION_MS = 120_000  # 2 minutes


def build_turns(segments: list[dict]) -> list[Turn]:
    """
    Input: list of dicts from adapter [{"speaker", "text", "start_ms", "end_ms"}]
    Output: list of Turn objects, one per contiguous speaker block.
    Consecutive segments by the same speaker are merged ONLY if:
      1. The gap between them is less than MAX_MERGE_GAP_MS (2 seconds)
      2. The merged turn duration stays under MAX_TURN_DURATION_MS (2 minutes)
    Do NOT filter short turns. Keep everything including single-word turns.
    """
    turns = []
    if not segments:
        return turns

    current_speaker = segments[0]["speaker"]
    current_texts = [segments[0]["text"]]
    current_start = segments[0]["start_ms"]
    current_end = segments[0]["end_ms"]

    for seg in segments[1:]:
        same_speaker = seg["speaker"] == current_speaker
        gap = seg["start_ms"] - current_end if same_speaker else 0
        merged_duration = seg["end_ms"] - current_start if same_speaker else 0

        if same_speaker and gap < MAX_MERGE_GAP_MS and merged_duration < MAX_TURN_DURATION_MS:
            current_texts.append(seg["text"])
            current_end = seg["end_ms"]
        else:
            turns.append(Turn(
                index=len(turns),
                speaker=current_speaker,
                text=" ".join(current_texts),
                start_ms=current_start,
                end_ms=current_end,
            ))
            current_speaker = seg["speaker"]
            current_texts = [seg["text"]]
            current_start = seg["start_ms"]
            current_end = seg["end_ms"]

    # Don't forget the last turn
    turns.append(Turn(
        index=len(turns),
        speaker=current_speaker,
        text=" ".join(current_texts),
        start_ms=current_start,
        end_ms=current_end,
    ))

    # Diagnostic: show merge stats
    if len(segments) > 1:
        same_speaker_gaps = []
        duration_splits = 0
        for i in range(1, len(segments)):
            if segments[i]["speaker"] == segments[i-1]["speaker"]:
                gap = segments[i]["start_ms"] - segments[i-1]["end_ms"]
                same_speaker_gaps.append(gap)
        if same_speaker_gaps:
            merged = sum(1 for g in same_speaker_gaps if g < MAX_MERGE_GAP_MS)
            split = sum(1 for g in same_speaker_gaps if g >= MAX_MERGE_GAP_MS)
            avg_gap = sum(same_speaker_gaps) / len(same_speaker_gaps)
            print(f"  Turn builder: {len(segments)} segments → {len(turns)} turns")
            print(f"    Same-speaker consecutive pairs: {len(same_speaker_gaps)}")
            print(f"    Merged (gap < {MAX_MERGE_GAP_MS}ms): {merged}")
            print(f"    Split (gap >= {MAX_MERGE_GAP_MS}ms): {split}")
            print(f"    Max turn duration cap: {MAX_TURN_DURATION_MS // 1000}s")
            print(f"    Avg same-speaker gap: {avg_gap:.0f}ms")
        else:
            print(f"  Turn builder: {len(segments)} segments → {len(turns)} turns (no consecutive same-speaker pairs)")

    return turns
