from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

from app.ingestion import RecordingFile
from app.transcription import TranscriptionResult
from app.schemas import CallMetadata, CallAnalysis


def ensure_calls_dir(base_data_dir: str | Path = "data") -> Path:
    """
    Ensure that the data/calls directory exists and return its Path.
    """
    base = Path(base_data_dir)
    calls_dir = base / "calls"
    calls_dir.mkdir(parents=True, exist_ok=True)
    return calls_dir


def generate_call_id(recording: RecordingFile) -> str:
    """
    Generate a simple call_id based on the filename and its modified timestamp.
    Example: conversation_1732041234
    """
    stem = recording.path.stem
    ts = int(recording.modified_time)
    return f"{stem}_{ts}"


def build_call_metadata(
    recording: RecordingFile,
    client_id: str,
) -> CallMetadata:
    """
    Build a CallMetadata object from a RecordingFile.
    For now, we only know basic file info; other fields stay None.
    """
    call_id = generate_call_id(recording)
    return CallMetadata(
        call_id=call_id,
        client_id=client_id,
        audio_file=str(recording.path),
        # You can later populate duration_seconds, agent_name, etc.
        extra_metadata={
            "size_bytes": recording.size_bytes,
            "modified_time": recording.modified_time,
            "created_at": datetime.utcnow().isoformat(),
        },
    )


def build_call_analysis_from_transcription(
    result: TranscriptionResult,
    client_id: str,
) -> CallAnalysis:
    """
    Build a minimal CallAnalysis object from a TranscriptionResult.
    At this stage, only metadata + transcript are filled.
    All AI analysis fields (sentiment, quality, etc.) remain None.
    """
    metadata = build_call_metadata(result.recording, client_id=client_id)
    return CallAnalysis(
        metadata=metadata,
        transcript=result.text,
        # sentiment / intent / quality / etc. will be filled later stages
        raw_llm_outputs={},
    )


def save_call_analysis(
    call_analysis: CallAnalysis,
    base_data_dir: str | Path = "data",
) -> Path:
    """
    Save a single CallAnalysis object as JSON under data/calls/.

    Returns:
        Path to the saved JSON file.
    """
    calls_dir = ensure_calls_dir(base_data_dir)
    call_id = call_analysis.metadata.call_id

    out_path = calls_dir / f"{call_id}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            call_analysis.model_dump(mode="json"),
            f,
            ensure_ascii=False,
            indent=2,
        )

    return out_path


def save_transcription_result(
    result: TranscriptionResult,
    client_id: str,
    base_data_dir: str | Path = "data",
) -> Path:
    """
    Backwards-compatible helper:
    Take a TranscriptionResult, wrap it into a CallAnalysis, and save it.
    """
    call_analysis = build_call_analysis_from_transcription(result, client_id=client_id)
    return save_call_analysis(call_analysis, base_data_dir=base_data_dir)


def save_transcription_results(
    results: Iterable[TranscriptionResult],
    client_id: str,
    base_data_dir: str | Path = "data",
) -> List[Path]:
    """
    Save multiple TranscriptionResult objects as CallAnalysis JSON files.

    Returns:
        List of Paths to the saved JSON files.
    """
    paths: List[Path] = []
    for r in results:
        path = save_transcription_result(r, client_id=client_id, base_data_dir=base_data_dir)
        paths.append(path)
    return paths
