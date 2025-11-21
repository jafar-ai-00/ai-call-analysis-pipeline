from __future__ import annotations

import json
from pathlib import Path
from typing import List

from app.schemas import CallAnalysis
from app.analysis_sentiment import analyze_sentiment_for_call
from app.analysis_intent_topics import analyze_intent_topics_for_call
from app.analysis_quality import analyze_quality_for_call
from app.analysis_compliance import analyze_compliance_for_call
from app.analysis_outcome import analyze_outcome_for_call






def list_call_files(data_dir: str | Path = "data") -> List[Path]:
    """
    List all JSON call files under data/calls.
    """
    base = Path(data_dir)
    calls_dir = base / "calls"
    if not calls_dir.exists():
        return []
    return sorted(calls_dir.glob("*.json"))


def load_call(path: Path) -> CallAnalysis:
    """
    Load a CallAnalysis object from a JSON file.
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return CallAnalysis.model_validate(data)


def save_call(path: Path, call: CallAnalysis) -> None:
    """
    Save a CallAnalysis object back to a JSON file.
    """
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            call.model_dump(mode="json"),
            f,
            ensure_ascii=False,
            indent=2,
        )


def run_sentiment_for_all_calls(
    data_dir: str | Path = "data",
    model: str = "gpt-4o",
) -> None:
    """
    For every call JSON in data/calls, run sentiment analysis and
    update the 'sentiment' and 'raw_llm_outputs["sentiment"]' fields.
    """
    call_paths = list_call_files(data_dir=data_dir)
    if not call_paths:
        print("No call JSON files found in data/calls.")
        return

    print(f"Found {len(call_paths)} call file(s) for sentiment analysis.")

    for path in call_paths:
        print(f"\n📄 Processing: {path.name}")
        call = load_call(path)

        # Skip if sentiment already exists (optional behavior)
        if call.sentiment is not None:
            print("   🔁 Sentiment already present, skipping.")
            continue

        sentiment_model, raw_json = analyze_sentiment_for_call(call, model=model)
        call.sentiment = sentiment_model

        # Keep raw LLM output for debugging / later tuning
        if call.raw_llm_outputs is None:
            call.raw_llm_outputs = {}
        call.raw_llm_outputs["sentiment"] = raw_json

        save_call(path, call)
        print("   ✅ Sentiment analysis saved.")


def run_intent_topics_for_all_calls(
    data_dir: str | Path = "data",
    model: str = "gpt-4o",
) -> None:
    """
    For every call JSON in data/calls, run intent/topics analysis and
    update the 'intent_and_topics' and 'raw_llm_outputs["intent_and_topics"]' fields.
    """
    call_paths = list_call_files(data_dir=data_dir)
    if not call_paths:
        print("No call JSON files found in data/calls.")
        return

    print(f"Found {len(call_paths)} call file(s) for intent/topics analysis.")

    for path in call_paths:
        print(f"\n📄 Processing: {path.name}")
        call = load_call(path)

        # Skip if intent_and_topics already exists (optional behavior)
        if call.intent_and_topics is not None:
            print("   🔁 Intent/topics already present, skipping.")
            continue

        intent_model, raw_json = analyze_intent_topics_for_call(call, model=model)
        call.intent_and_topics = intent_model

        # Keep raw LLM output for debugging / later tuning
        if call.raw_llm_outputs is None:
            call.raw_llm_outputs = {}
        call.raw_llm_outputs["intent_and_topics"] = raw_json

        save_call(path, call)
        print("   ✅ Intent & topics analysis saved.")

def run_quality_for_all_calls(
    data_dir: str | Path = "data",
    model: str = "gpt-4o",
) -> None:
    """
    For every call JSON in data/calls, run call quality analysis and
    update the 'call_quality' and 'raw_llm_outputs["call_quality"]' fields.
    """
    call_paths = list_call_files(data_dir=data_dir)
    if not call_paths:
        print("No call JSON files found in data/calls.")
        return

    print(f"Found {len(call_paths)} call file(s) for quality analysis.")

    for path in call_paths:
        print(f"\n📄 Processing: {path.name}")
        call = load_call(path)

        # Skip if call_quality already exists (optional behavior)
        if call.call_quality is not None:
            print("   🔁 Call quality already present, skipping.")
            continue

        quality_model, raw_json = analyze_quality_for_call(call, model=model)
        call.call_quality = quality_model

        # Keep raw LLM output for debugging / later tuning
        if call.raw_llm_outputs is None:
            call.raw_llm_outputs = {}
        call.raw_llm_outputs["call_quality"] = raw_json

        save_call(path, call)
        print("   ✅ Call quality analysis saved.")

def run_compliance_for_all_calls(
    data_dir: str | Path = "data",
    model: str = "gpt-4o",
    required_phrases: list[str] | None = None,
    forbidden_phrases: list[str] | None = None,
) -> None:
    """
    For every call JSON in data/calls, run compliance & risk analysis and
    update the 'compliance_and_risk' and 'raw_llm_outputs["compliance_and_risk"]' fields.
    """
    call_paths = list_call_files(data_dir=data_dir)
    if not call_paths:
        print("No call JSON files found in data/calls.")
        return

    if required_phrases is None:
        required_phrases = []
    if forbidden_phrases is None:
        forbidden_phrases = []

    print(f"Found {len(call_paths)} call file(s) for compliance analysis.")

    for path in call_paths:
        print(f"\n📄 Processing: {path.name}")
        call = load_call(path)

        # Skip if already present (optional behavior)
        if call.compliance_and_risk is not None:
            print("   🔁 Compliance & risk already present, skipping.")
            continue

        compliance_model, raw_json = analyze_compliance_for_call(
            call,
            required_phrases=required_phrases,
            forbidden_phrases=forbidden_phrases,
            model=model,
        )
        call.compliance_and_risk = compliance_model

        if call.raw_llm_outputs is None:
            call.raw_llm_outputs = {}
        call.raw_llm_outputs["compliance_and_risk"] = raw_json

        save_call(path, call)
        print("   ✅ Compliance & risk analysis saved.")
        

def run_outcome_for_all_calls(
    data_dir: str | Path = "data",
    model: str = "gpt-4o",
) -> None:
    """
    For every call JSON in data/calls, run outcome & follow-up analysis and
    update the 'outcome_and_followup' and 'raw_llm_outputs["outcome_and_followup"]' fields.
    """
    call_paths = list_call_files(data_dir=data_dir)
    if not call_paths:
        print("No call JSON files found in data/calls.")
        return

    print(f"Found {len(call_paths)} call file(s) for outcome analysis.")

    for path in call_paths:
        print(f"\n📄 Processing: {path.name}")
        call = load_call(path)

        # Skip if already present (optional behavior)
        if call.outcome_and_followup is not None:
            print("   🔁 Outcome & follow-up already present, skipping.")
            continue

        outcome_model, raw_json = analyze_outcome_for_call(call, model=model)
        call.outcome_and_followup = outcome_model

        if call.raw_llm_outputs is None:
            call.raw_llm_outputs = {}
        call.raw_llm_outputs["outcome_and_followup"] = raw_json

        save_call(path, call)
        print("   ✅ Outcome & follow-up analysis saved.")
