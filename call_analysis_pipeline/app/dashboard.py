from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import List, Dict, Optional

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.schemas import (
    CallAnalysis,
    SentimentLabel,
    RiskLevel,
)
from app.vectorstore import semantic_search_calls



DATA_DIR = Path("data")
CALLS_DIR = DATA_DIR / "calls"


# =========================
# Data loading & stats
# =========================

def load_calls() -> List[CallAnalysis]:
    """
    Load all CallAnalysis JSON files from data/calls.
    """
    if not CALLS_DIR.exists():
        return []

    calls: List[CallAnalysis] = []
    for path in sorted(CALLS_DIR.glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            call = CallAnalysis.model_validate(data)
            calls.append(call)
        except Exception as e:
            # In a real app you might log this instead
            st.warning(f"Failed to load {path.name}: {e}")
    return calls


def compute_basic_stats(calls: List[CallAnalysis]) -> Dict[str, Optional[float]]:
    """
    Compute simple aggregate stats across calls.
    """
    stats = {
        "total_calls": len(calls),
        "sentiment_counts": {
            "positive": 0,
            "neutral": 0,
            "negative": 0,
            "unknown": 0,
        },
        "average_quality_score": None,
        "risk_counts": {
            "low": 0,
            "medium": 0,
            "high": 0,
            "critical": 0,
            "unknown": 0,
        },
    }

    if not calls:
        return stats

    quality_scores = []

    for call in calls:
        # Sentiment
        if call.sentiment and call.sentiment.overall:
            label = call.sentiment.overall.value
            if label in stats["sentiment_counts"]:
                stats["sentiment_counts"][label] += 1
            else:
                stats["sentiment_counts"]["unknown"] += 1
        else:
            stats["sentiment_counts"]["unknown"] += 1

        # Quality
        if call.call_quality and call.call_quality.overall_quality_score is not None:
            quality_scores.append(call.call_quality.overall_quality_score)

        # Risk
        if call.compliance_and_risk and call.compliance_and_risk.risk_level:
            risk_label = call.compliance_and_risk.risk_level.value
            if risk_label in stats["risk_counts"]:
                stats["risk_counts"][risk_label] += 1
            else:
                stats["risk_counts"]["unknown"] += 1
        else:
            stats["risk_counts"]["unknown"] += 1

    if quality_scores:
        stats["average_quality_score"] = sum(quality_scores) / len(quality_scores)

    return stats


# =========================
# UI helpers
# =========================

def render_summary_section(stats: Dict[str, Optional[float]]) -> None:
    st.subheader("Summary")

    total_calls = stats["total_calls"]
    sentiment_counts = stats["sentiment_counts"]
    risk_counts = stats["risk_counts"]
    avg_quality = stats["average_quality_score"]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total calls", total_calls)

    with col2:
        if avg_quality is not None:
            st.metric("Avg quality score", f"{avg_quality:.1f} / 100")
        else:
            st.metric("Avg quality score", "N/A")

    with col3:
        high_risk = risk_counts.get("high", 0) + risk_counts.get("critical", 0)
        st.metric("High / critical risk", high_risk)

    st.markdown("#### Sentiment distribution")
    st.write(
        {
            "positive": sentiment_counts["positive"],
            "neutral": sentiment_counts["neutral"],
            "negative": sentiment_counts["negative"],
            "unknown": sentiment_counts["unknown"],
        }
    )

    st.markdown("#### Risk distribution")
    st.write(risk_counts)


def render_call_card(call: CallAnalysis) -> None:
    """
    Render a single call in an expandable card.
    """
    meta = call.metadata
    call_id = meta.call_id
    audio_file = meta.audio_file

    # Safe access helpers
    sentiment_label = call.sentiment.overall.value if call.sentiment else "unknown"
    quality_score = (
        call.call_quality.overall_quality_score
        if call.call_quality and call.call_quality.overall_quality_score is not None
        else "N/A"
    )
    risk_label = (
        call.compliance_and_risk.risk_level.value
        if call.compliance_and_risk and call.compliance_and_risk.risk_level
        else "unknown"
    )
    primary_intent = (
        call.intent_and_topics.primary_intent
        if call.intent_and_topics and call.intent_and_topics.primary_intent
        else "N/A"
    )
    resolution_status = (
        call.outcome_and_followup.resolution_status.value
        if call.outcome_and_followup and call.outcome_and_followup.resolution_status
        else "N/A"
    )

    # Short transcript preview
    transcript_preview = (call.transcript or "")[:220].replace("\n", " ")
    if len(call.transcript or "") > 220:
        transcript_preview += "..."

    with st.expander(f"Call: {call_id}  |  Sentiment: {sentiment_label}  |  Risk: {risk_label}"):
        st.markdown(f"**Audio file:** `{audio_file}`")
        st.markdown(f"**Primary intent:** {primary_intent}")
        st.markdown(f"**Quality score:** {quality_score}")
        st.markdown(f"**Resolution:** {resolution_status}")

        st.markdown("**Transcript preview:**")
        st.write(transcript_preview)

        # Optional deeper details
        with st.expander("View full transcript"):
            st.text(call.transcript)

        with st.expander("View sentiment details"):
            st.json(call.sentiment.model_dump(mode='json') if call.sentiment else {})

        with st.expander("View intent & topics"):
            st.json(call.intent_and_topics.model_dump(mode='json') if call.intent_and_topics else {})

        with st.expander("View call quality breakdown"):
            st.json(call.call_quality.model_dump(mode='json') if call.call_quality else {})

        with st.expander("View compliance & risk details"):
            st.json(call.compliance_and_risk.model_dump(mode='json') if call.compliance_and_risk else {})

        with st.expander("View outcome & follow-up"):
            st.json(call.outcome_and_followup.model_dump(mode='json') if call.outcome_and_followup else {})

def render_semantic_search_section() -> None:
    st.markdown("## 🔍 Semantic Search across calls")

    with st.expander("Open semantic search"):
        query = st.text_input(
            "Enter a search query (semantic)",
            value="",
            key="semantic_query",
            placeholder="e.g. angry customer asking about refund",
        )
        n_results = st.slider(
            "Number of results",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
            key="semantic_n_results",
        )

        run_search = st.button("Search", key="semantic_search_btn")

        if run_search:
            if not query.strip():
                st.warning("Please enter a query before searching.")
                return

            with st.spinner("Running semantic search..."):
                try:
                    matches = semantic_search_calls(
                        query=query,
                        n_results=n_results,
                        chroma_db_dir="chroma_db",
                        collection_name="calls",
                    )
                except Exception as e:
                    st.error(f"Error running semantic search: {e}")
                    return

            if not matches:
                st.info("No matches found.")
                return

            st.markdown(f"### Top {len(matches)} result(s)")
            for i, m in enumerate(matches, start=1):
                meta = m.get("metadata") or {}
                call_id = meta.get("call_id")
                client_id = meta.get("client_id")
                sentiment = meta.get("sentiment") or "unknown"
                risk_level = meta.get("risk_level") or "unknown"
                quality_score = meta.get("quality_score")

                doc = m.get("document") or ""
                snippet = doc[:220].replace("\n", " ")
                if len(doc) > 220:
                    snippet += "..."

                distance = m.get("distance")

                with st.expander(
                    f"Result {i}: call={call_id} | sentiment={sentiment} | risk={risk_level}"
                ):
                    st.markdown(f"**Client ID:** `{client_id}`")
                    st.markdown(f"**Quality score:** `{quality_score}`")
                    st.markdown(f"**Distance:** `{distance}`")

                    st.markdown("**Matching text snippet:**")
                    st.write(snippet)

                    st.markdown("**Raw metadata:**")
                    st.json(meta)

# =========================
# Main app
# =========================

def main():
    st.set_page_config(
        page_title="Call Analysis Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("📞 Call Analysis Dashboard")
    st.caption("AI-powered analysis of daily calls for a single client.")

    calls = load_calls()

    if not calls:
        st.warning("No calls found in `data/calls`. Run your pipeline first to generate analyzed calls.")
        return

    stats = compute_basic_stats(calls)

    # Sidebar filters
    st.sidebar.header("Filters")

    sentiment_filter = st.sidebar.multiselect(
        "Sentiment",
        options=["positive", "neutral", "negative", "unknown"],
        default=["positive", "neutral", "negative", "unknown"],
    )

    risk_filter = st.sidebar.multiselect(
        "Risk level",
        options=["low", "medium", "high", "critical", "unknown"],
        default=["low", "medium", "high", "critical", "unknown"],
    )

    min_quality = st.sidebar.slider(
        "Minimum quality score",
        min_value=0,
        max_value=100,
        value=0,
        step=5,
    )

    # Filter calls in-memory
    filtered_calls: List[CallAnalysis] = []
    for call in calls:
        # Sentiment filter
        s_label = call.sentiment.overall.value if call.sentiment else "unknown"
        if s_label not in sentiment_filter:
            continue

        # Risk filter
        r_label = (
            call.compliance_and_risk.risk_level.value
            if call.compliance_and_risk and call.compliance_and_risk.risk_level
            else "unknown"
        )
        if r_label not in risk_filter:
            continue

        # Quality filter
        q_score = (
            call.call_quality.overall_quality_score
            if call.call_quality and call.call_quality.overall_quality_score is not None
            else None
        )
        if q_score is not None and q_score < min_quality:
            continue

        filtered_calls.append(call)

    # Main layout
    render_summary_section(stats)

    render_semantic_search_section()

    st.markdown("## Calls")
    st.caption(f"Showing {len(filtered_calls)} of {len(calls)} calls after filters.")

    if not filtered_calls:
        st.info("No calls match the current filters.")
        return

    for call in filtered_calls:
        render_call_card(call)


if __name__ == "__main__":
    main()
