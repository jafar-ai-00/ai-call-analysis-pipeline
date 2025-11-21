from __future__ import annotations

import json
from typing import Optional, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from app.schemas import (
    CallAnalysis,
    OutcomeFollowupAnalysis,
)


SYSTEM_PROMPT = """
You are an AI assistant that analyzes customer support calls.

CRITICAL INSTRUCTIONS:
- You ALWAYS respond with a single valid JSON object.
- Do NOT include any explanations, markdown, backticks, or surrounding text.
- Do NOT include trailing commas.
- If you are unsure about a value, use null or an empty list [] as appropriate.
- Follow exactly the field names, types, and allowed values described in the user message.
""".strip()


HUMAN_PROMPT = """
You are performing OUTCOME AND FOLLOW-UP ANALYSIS for a single customer call.

You must respond with a SINGLE valid JSON object with exactly these fields:

- resolution_status: one of "resolved", "partially_resolved", "unresolved"
- final_outcome: string or null
- followup_actions: array of objects, each with:
    - description: string
    - owner: string or null (e.g. "agent", "customer", "backoffice", "finance_team")
    - due_date: string in "YYYY-MM-DD" format or null
- escalation_required: true or false
- escalation_reason: string or null
- notes: string or null

Guidelines:
- "resolution_status":
  - "resolved": the customer's main issue was clearly addressed.
  - "partially_resolved": some progress but still pending.
  - "unresolved": the main issue remains unaddressed.
- "final_outcome": a short label such as:
  - "appointment_booked", "appointment_rescheduled", "appointment_cancelled",
  - "information_provided", "refund_approved", "refund_declined",
  - "issue_escalated", "no_clear_outcome", etc.
- "followup_actions": 0–5 items with specific actions (e.g. "Send SMS confirmation for rescheduled appointment").
  - "owner": who should do it (e.g. "agent", "customer", "backoffice").
  - If no clear due date, set due_date to null.
- "escalation_required": true if the case clearly needs a more senior person/department.
- "escalation_reason": short explanation if escalation_required is true.
- "notes": 1–3 sentences summarizing the outcome and next steps in natural language.

Remember:
- Do NOT add extra top-level fields.
- Do NOT include explanations or markdown.
- The output must be a single JSON object.

Call metadata (JSON):
{call_metadata_json}

Transcript:
\"\"\"{transcript}\"\"\"
""".strip()


def get_outcome_llm(model: str = "gpt-4o", temperature: float = 0.0) -> ChatOpenAI:
    """
    Create a ChatOpenAI instance for outcome & follow-up analysis.
    """
    return ChatOpenAI(model=model, temperature=temperature)


def build_outcome_prompt() -> ChatPromptTemplate:
    """
    Build the LangChain ChatPromptTemplate for outcome & follow-up analysis.
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ]
    )


def analyze_outcome_for_call(
    call: CallAnalysis,
    llm: Optional[ChatOpenAI] = None,
    model: str = "gpt-4o",
) -> Tuple[OutcomeFollowupAnalysis, dict]:
    """
    Run outcome & follow-up analysis for a single CallAnalysis instance.

    Returns:
        (outcome_model, raw_json_dict)
    """
    if llm is None:
        llm = get_outcome_llm(model=model)

    prompt = build_outcome_prompt()

    call_metadata_json = json.dumps(
        call.metadata.model_dump(mode="json"),
        ensure_ascii=False,
    )

    chain = prompt | llm

    response = chain.invoke(
        {
            "call_metadata_json": call_metadata_json,
            "transcript": call.transcript,
        }
    )

    content = response.content
    if not isinstance(content, str):
        raise ValueError(f"Expected string content from LLM, got: {type(content)}")

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM JSON response: {e}\nRaw content: {content}") from e

    outcome = OutcomeFollowupAnalysis.model_validate(data)

    return outcome, data
