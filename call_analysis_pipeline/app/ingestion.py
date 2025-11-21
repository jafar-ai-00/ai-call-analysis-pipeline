from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class RecordingFile:
    """
    Represents a single audio recording file discovered in the recordings directory.
    """
    path: Path
    name: str
    size_bytes: int
    modified_time: float  # POSIX timestamp (seconds since epoch)

    def pretty_size(self) -> str:
        """
        Human-readable file size, e.g. '1.2 MB'.
        """
        size = self.size_bytes
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} PB"


def discover_wav_recordings(recordings_dir: str | Path = "recordings") -> List[RecordingFile]:
    """
    Discover all .wav and .mp3 files under the given recordings directory (recursively).

    NOTE: Function name kept for backwards compatibility, but it now supports mp3 too.

    Args:
        recordings_dir: Root directory where audio files are stored.

    Returns:
        List of RecordingFile objects, sorted by modification time (oldest first).

    Raises:
        FileNotFoundError: If the recordings directory does not exist.
    """
    root = Path(recordings_dir)

    if not root.exists():
        raise FileNotFoundError(f"Recordings directory does not exist: {root.resolve()}")

    recordings: List[RecordingFile] = []
    allowed_exts = {".wav", ".mp3"}

    # Recursively search for audio files (case-insensitive)
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in allowed_exts:
            stat = path.stat()
            recordings.append(
                RecordingFile(
                    path=path,
                    name=path.name,
                    size_bytes=stat.st_size,
                    modified_time=stat.st_mtime,
                )
            )

    # Sort by modification time (oldest first) so processing is predictable
    recordings.sort(key=lambda r: r.modified_time)

    return recordings


def debug_print_recordings(recordings: List[RecordingFile]) -> None:
    """
    Utility function to print a simple table of discovered recordings.
    Useful while developing and testing ingestion.
    """
    if not recordings:
        print("No .wav or .mp3 recordings found.")
        return

    print(f"Discovered {len(recordings)} audio recording(s):")
    for rec in recordings:
        print(f"- {rec.name} | {rec.pretty_size()} | {rec.path}")
