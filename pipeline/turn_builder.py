from models import Turn


def build_turns(segments: list[dict]) -> list[Turn]:
    """
    Input: list of dicts from adapter [{"speaker", "text", "start_ms", "end_ms"}]
    Output: list of Turn objects, one per contiguous speaker block.
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
        if seg["speaker"] == current_speaker:
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
    return turns
