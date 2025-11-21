from __future__ import annotations

import os
import yaml
from pathlib import Path

from app.ingestion import discover_wav_recordings, debug_print_recordings
from app.transcription import transcribe_recordings
from app.storage import save_transcription_results
from app.analysis_runner import (
    run_sentiment_for_all_calls,
    run_intent_topics_for_all_calls,
    run_quality_for_all_calls,
    run_compliance_for_all_calls,
    run_outcome_for_all_calls,
)

CONFIG_PATH = Path("config/config.yaml")


def load_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found at {CONFIG_PATH}")
    with CONFIG_PATH.open("r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    client_id = config.get("client_id", "client_123")
    recordings_dir = config.get("recordings_dir", "recordings/")
    data_dir = config.get("data_dir", "data/")
    openai_conf = config.get("openai", {}) or {}
    whisper_model = openai_conf.get("whisper_model", "whisper-1")
    llm_model = openai_conf.get("llm_model", "gpt-4o")

    # Optional compliance config
    compliance_conf = config.get("compliance", {}) or {}
    required_phrases = compliance_conf.get("required_phrases", [])
    forbidden_phrases = compliance_conf.get("forbidden_phrases", [])

    print("=== AI Call Analysis: Full Pipeline ===")
    print(f"Client ID: {client_id}")
    print(f"Recordings dir: {recordings_dir}")
    print(f"Data dir: {data_dir}")
    print(f"Whisper model: {whisper_model}")
    print(f"LLM model: {llm_model}")

    # Check API key
    openai_env_var = openai_conf.get("api_key_env", "OPENAI_API_KEY")
    has_key = os.getenv(openai_env_var) is not None
    print(f"OpenAI API key env var expected: {openai_env_var}")
    print(f"OpenAI API key set in env? {'✅ Yes' if has_key else '❌ No'}")
    if not has_key:
        print("❌ Please export your OpenAI API key before running:")
        print(f"   export {openai_env_var}='sk-...' ")
        return

    # 1) Discover recordings
    print("\n[1/5] Discovering .wav recordings...")
    recordings = discover_wav_recordings(recordings_dir)
    debug_print_recordings(recordings)

    if not recordings:
        print("No recordings found. Put .wav files into the recordings/ folder.")
        return

    # 2) Transcribe with Whisper
    print("\n[2/5] Transcribing recordings with Whisper...")
    results = transcribe_recordings(recordings, model=whisper_model)

    # 3) Save minimal CallAnalysis JSON
    print("\n[3/5] Saving transcripts as CallAnalysis JSON...")
    saved_paths = save_transcription_results(
        results,
        client_id=client_id,
        base_data_dir=data_dir,
    )
    print(f"Saved {len(saved_paths)} call file(s) under {data_dir}/calls:")
    for p in saved_paths:
        print(f"- {p}")

    # 4) Run all analysis stages on data/calls/*.json
    print("\n[4/5] Running per-call analyses on existing JSON files...")

    print("   → Sentiment & emotion")
    run_sentiment_for_all_calls(data_dir=data_dir, model=llm_model)

    print("   → Intent & topics")
    run_intent_topics_for_all_calls(data_dir=data_dir, model=llm_model)

    print("   → Call quality & agent performance")
    run_quality_for_all_calls(data_dir=data_dir, model=llm_model)

    print("   → Compliance & risk")
    run_compliance_for_all_calls(
        data_dir=data_dir,
        model=llm_model,
        required_phrases=required_phrases,
        forbidden_phrases=forbidden_phrases,
    )

    print("   → Outcome & follow-up")
    run_outcome_for_all_calls(data_dir=data_dir, model=llm_model)

    print("\n[5/5] Done ✅")
    print("All calls in data/calls/ now have full analysis attached.")
    print("Open the dashboard with:")
    print("   uv run streamlit run app/dashboard.py")


if __name__ == "__main__":
    main()
