from pathlib import Path
import os
import yaml

from app.ingestion import discover_wav_recordings, debug_print_recordings
from app.transcription import transcribe_recordings
from app.storage import save_transcription_results

CONFIG_PATH = Path("config/config.yaml")


def load_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found at {CONFIG_PATH}")
    with CONFIG_PATH.open("r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    print("✅ Call Analysis Pipeline environment is ready.")
    print(f"Client ID: {config.get('client_id')}")
    print(f"Recordings dir: {config.get('recordings_dir')}")

    openai_env_var = config["openai"]["api_key_env"]
    has_key = os.getenv(openai_env_var) is not None
    print(f"OpenAI API key env var expected: {openai_env_var}")
    print(f"OpenAI API key set in env? {'✅ Yes' if has_key else '❌ No'}")

    # Discover recordings
    recordings_dir = config.get("recordings_dir", "recordings/")
    recordings = discover_wav_recordings(recordings_dir)
    debug_print_recordings(recordings)

    if not recordings:
        print("No recordings to transcribe. Put some .wav files in the recordings/ folder.")
        return

    # Transcribe
    whisper_model = config["openai"].get("whisper_model", "whisper-1")
    results = transcribe_recordings(recordings, model=whisper_model)

    # (Optional) print preview
    for r in results:
        preview = r.text[:200].replace("\n", " ")
        print(f"\n--- Transcript preview for {r.recording.name} ---")
        print(preview)
        print("...")

    # ✅ Save transcripts to data/calls/<call_id>.json
    data_dir = config.get("data_dir", "data/")
    client_id = config.get("client_id", "client_123")
    saved_paths = save_transcription_results(results, client_id=client_id, base_data_dir=data_dir)

    print(f"\n💾 Saved {len(saved_paths)} transcript file(s):")
    for p in saved_paths:
        print(f"- {p}")


if __name__ == "__main__":
    main()
