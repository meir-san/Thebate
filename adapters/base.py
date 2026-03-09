from abc import ABC, abstractmethod


class BaseAdapter(ABC):
    @abstractmethod
    def transcribe(self, audio_path: str) -> list[dict]:
        """
        Transcribe and diarize audio file.
        Returns list of dicts: [{"speaker": str, "text": str, "start_ms": int, "end_ms": int}]
        Speaker labels must be normalized to "SPEAKER_00", "SPEAKER_01" etc.
        """
        pass
