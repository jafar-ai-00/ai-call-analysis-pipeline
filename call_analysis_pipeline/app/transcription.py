from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from openai import OpenAI

from app.ingestion import RecordingFile


@dataclass
class TranscriptionResult:
    """
    Result of transcribing a single audio recording.
    """
    recording: RecordingFile
    text: str
    language: Optional[str] = None


def get_openai_client() -> OpenAI:
    """
    Create an OpenAI client using the API key from the environment.
    """
    return OpenAI()


def transcribe_recording(
    recording: RecordingFile,
    model: str = "whisper-1",
    client: Optional[OpenAI] = None,
) -> TranscriptionResult:
    """
    Transcribe a single audio recording using OpenAI Whisper API.

    Args:
        recording: RecordingFile object describing the .wav file.
        model: Whisper model name (default 'whisper-1').
        client: Optional shared OpenAI client.

    Returns:
        TranscriptionResult with transcript text and detected language (if available).
    """
    if client is None:
        client = get_openai_client()

    path: Path = recording.path

    with path.open("rb") as audio_file:
        response = client.audio.transcriptions.create(
            model=model,
            file=audio_file,
            response_format="json",
        )

    # The new OpenAI SDK returns an object with 'text' and possibly 'language'
    text: str = response.text
    language: Optional[str] = getattr(response, "language", None)

    return TranscriptionResult(
        recording=recording,
        text=text,
        language=language,
    )


def transcribe_recordings(
    recordings: Iterable[RecordingFile],
    model: str = "whisper-1",
    client: Optional[OpenAI] = None,
) -> List[TranscriptionResult]:
    """
    Transcribe multiple recordings sequentially.

    Args:
        recordings: Iterable of RecordingFile objects.
        model: Whisper model name.
        client: Optional shared OpenAI client.

    Returns:
        List of TranscriptionResult objects.
    """
    if client is None:
        client = get_openai_client()

    results: List[TranscriptionResult] = []

    for rec in recordings:
        print(f"🎧 Transcribing: {rec.name} ({rec.pretty_size()})")
        try:
            result = transcribe_recording(rec, model=model, client=client)
            print(f"✅ Done: {rec.name} | transcript length: {len(result.text)} chars")
            if result.language:
                print(f"   Detected language: {result.language}")
            results.append(result)
        except Exception as e:
            print(f"❌ Failed to transcribe {rec.name}: {e}")

    return results
