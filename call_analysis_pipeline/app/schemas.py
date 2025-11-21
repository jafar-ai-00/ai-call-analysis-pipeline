from __future__ import annotations

from datetime import datetime, date
from enum import Enum
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field


# =========================
# Enums
# =========================

class SentimentLabel(str, Enum):
    positive = "positive"
    neutral = "neutral"
    negative = "negative"


class RiskLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class ResolutionStatus(str, Enum):
    resolved = "resolved"
    partially_resolved = "partially_resolved"
    unresolved = "unresolved"


# =========================
# Call Metadata
# =========================

class CallMetadata(BaseModel):
    call_id: str = Field(..., description="Internal ID for this call")
    client_id: str = Field(..., description="ID of the client")
    audio_file: str = Field(..., description="Path or URI to the audio file")

    call_start_time: Optional[datetime] = Field(
        default=None,
        description="Start time of the call, if known."
    )
    call_end_time: Optional[datetime] = Field(
        default=None,
        description="End time of the call, if known."
    )
    duration_seconds: Optional[float] = Field(
        default=None,
        description="Duration of the call in seconds."
    )

    agent_name: Optional[str] = Field(
        default=None,
        description="Name or ID of the agent handling the call."
    )
    customer_phone: Optional[str] = Field(
        default=None,
        description="Caller phone number (store masked if possible)."
    )

    extra_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Any additional metadata (queue, campaign, language, etc.)."
    )


# =========================
# Sentiment & Emotion
# =========================

class SentimentSegment(BaseModel):
    segment_label: str = Field(..., description="E.g. 'intro', 'problem_explained'.")
    start_second: Optional[float] = Field(default=None)
    end_second: Optional[float] = Field(default=None)
    sentiment: SentimentLabel = Field(..., description="Segment-level sentiment.")
    notes: Optional[str] = Field(default=None)


class SentimentAnalysis(BaseModel):
    overall: SentimentLabel = Field(..., description="Overall sentiment of the call.")
    score: Optional[float] = Field(
        default=None,
        description="Numeric sentiment score, e.g., -1.0 to +1.0."
    )
    emotion_tags: List[str] = Field(
        default_factory=list,
        description="High-level emotion tags, e.g., ['frustration', 'relief']."
    )
    sentiment_timeline: List[SentimentSegment] = Field(
        default_factory=list,
        description="Optional sentiment over segments of the call."
    )
    notes: Optional[str] = Field(default=None)


# =========================
# Intent & Topics
# =========================

class IntentTopicsAnalysis(BaseModel):
    primary_intent: Optional[str] = Field(
        default=None,
        description="Main reason for the call, e.g., 'reschedule_appointment'."
    )
    secondary_intents: List[str] = Field(
        default_factory=list,
        description="Other relevant intents."
    )
    topics: List[str] = Field(
        default_factory=list,
        description="Key topics discussed."
    )
    key_phrases: List[str] = Field(
        default_factory=list,
        description="Important phrases or concepts mentioned."
    )
    intent_confidence: Optional[float] = Field(
        default=None,
        description="Optional confidence score 0.0–1.0."
    )
    notes: Optional[str] = Field(default=None)


# =========================
# Call Quality / Agent Performance
# =========================

class CallQualityScores(BaseModel):
    greeting: Optional[int] = Field(default=None)
    listening_and_empathy: Optional[int] = Field(default=None)
    clarity_of_explanations: Optional[int] = Field(default=None)
    professionalism: Optional[int] = Field(default=None)
    script_adherence: Optional[int] = Field(default=None)


class CallQualityAnalysis(BaseModel):
    overall_quality_score: Optional[int] = Field(
        default=None,
        description="Overall call quality score (e.g., 0–100)."
    )
    scores: CallQualityScores = Field(
        default_factory=CallQualityScores,
        description="Sub-scores for different aspects."
    )
    strengths: List[str] = Field(
        default_factory=list,
        description="What the agent did well."
    )
    improvements: List[str] = Field(
        default_factory=list,
        description="Concrete suggestions for improvement."
    )
    notes: Optional[str] = Field(default=None)


# =========================
# Compliance & Risk
# =========================

class PIIRedaction(BaseModel):
    type: str = Field(..., description="E.g. 'phone_number', 'email', 'card_number'.")
    original_value: Optional[str] = Field(default=None)
    masked_value: Optional[str] = Field(default=None)


class ComplianceRiskAnalysis(BaseModel):
    required_phrases_present: List[str] = Field(default_factory=list)
    missing_required_phrases: List[str] = Field(default_factory=list)
    forbidden_phrases_detected: List[str] = Field(default_factory=list)
    pii_detected: List[PIIRedaction] = Field(default_factory=list)
    risk_level: RiskLevel = Field(default=RiskLevel.low)
    notes: Optional[str] = Field(default=None)


# =========================
# Outcome & Follow-up
# =========================

class FollowupAction(BaseModel):
    description: str = Field(..., description="Description of the follow-up action.")
    owner: Optional[str] = Field(
        default=None,
        description="Who is responsible, e.g. 'agent', 'customer', 'backoffice'."
    )
    due_date: Optional[date] = Field(default=None)


class OutcomeFollowupAnalysis(BaseModel):
    resolution_status: ResolutionStatus = Field(
        default=ResolutionStatus.unresolved,
        description="resolved / partially_resolved / unresolved."
    )
    final_outcome: Optional[str] = Field(default=None)
    followup_actions: List[FollowupAction] = Field(default_factory=list)
    escalation_required: bool = Field(default=False)
    escalation_reason: Optional[str] = Field(default=None)
    notes: Optional[str] = Field(default=None)


# =========================
# Custom / Business Metrics
# =========================

class CustomMetrics(BaseModel):
    appointment_booked: Optional[bool] = Field(default=None)
    appointment_rescheduled: Optional[bool] = Field(default=None)
    refund_requested: Optional[bool] = Field(default=None)
    upsell_attempted: Optional[bool] = Field(default=None)
    upsell_successful: Optional[bool] = Field(default=None)

    extra: Dict[str, Any] = Field(
        default_factory=dict,
        description="Any additional custom fields."
    )


# =========================
# Full Per-Call Analysis
# =========================

class CallAnalysis(BaseModel):
    metadata: CallMetadata = Field(..., description="Basic information about the call.")
    transcript: str = Field(..., description="Full transcript of the call.")

    sentiment: Optional[SentimentAnalysis] = Field(default=None)
    intent_and_topics: Optional[IntentTopicsAnalysis] = Field(default=None)
    call_quality: Optional[CallQualityAnalysis] = Field(default=None)
    compliance_and_risk: Optional[ComplianceRiskAnalysis] = Field(default=None)
    outcome_and_followup: Optional[OutcomeFollowupAnalysis] = Field(default=None)
    custom_metrics: Optional[CustomMetrics] = Field(default=None)

    raw_llm_outputs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional raw LLM responses for debugging."
    )


