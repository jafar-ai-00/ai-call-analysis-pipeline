from __future__ import annotations

import json
from typing import Optional, Tuple, List

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from app.schemas import (
    CallAnalysis,
    ComplianceRiskAnalysis,
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
You are performing COMPLIANCE AND RISK ANALYSIS for a single customer call.

You must respond with a SINGLE valid JSON object with exactly these fields:

- required_phrases_present: array of strings
- missing_required_phrases: array of strings
- forbidden_phrases_detected: array of strings
- pii_detected: array of objects, each with:
    - type: string (e.g. "phone_number", "email", "card_number", "address")
    - original_value: string or null
    - masked_value: string or null
- risk_level: one of "low", "medium", "high", "critical"
- notes: string or null

The client has the following compliance rules:

- REQUIRED_PHRASES: these should be said at least once during the call.
  Use the exact phrase from the list when reporting them, even if the call used a close paraphrase.
  REQUIRED_PHRASES (JSON): {required_phrases_json}

- FORBIDDEN_PHRASES: these should NOT be said.
  Use the exact phrase from the list when reporting them, even if the call used a close paraphrase.
  FORBIDDEN_PHRASES (JSON): {forbidden_phrases_json}

Guidelines:
- For required phrases:
  - If a phrase or a clear paraphrase appears, include it in required_phrases_present.
  - If a phrase does not appear, include it in missing_required_phrases.
- For forbidden phrases:
  - If a phrase or a clear paraphrase appears, include it in forbidden_phrases_detected.
- For PII:
  - Detect things like phone numbers, email addresses, payment card numbers, and physical addresses.
  - "type": label such as "phone_number", "email", "card_number", "address".
  - "masked_value": use a partially hidden version, e.g. "+9715XXXXXXX".
  - "original_value": can be null if you want to avoid storing the real value.
- risk_level:
  - "low": no significant issues.
  - "medium": minor missing required phrases or light PII risk.
  - "high": clear missing required phrases or use of forbidden phrases or sensitive PII.
  - "critical": severe compliance breach or potential legal exposure.
- notes: 1–3 sentences explaining why you chose that risk level.

Remember:
- Do NOT add extra top-level fields.
- Do NOT include explanations or markdown.
- The output must be a single JSON object.

Call metadata (JSON):
{call_metadata_json}

Transcript:
\"\"\"{transcript}\"\"\"
""".strip()


def get_compliance_llm(model: str = "gpt-4o", temperature: float = 0.0) -> ChatOpenAI:
    """
    Create a ChatOpenAI instance for compliance & risk analysis.
    """
    return ChatOpenAI(model=model, temperature=temperature)


def build_compliance_prompt() -> ChatPromptTemplate:
    """
    Build the LangChain ChatPromptTemplate for compliance & risk analysis.
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ]
    )


def analyze_compliance_for_call(
    call: CallAnalysis,
    required_phrases: Optional[List[str]] = None,
    forbidden_phrases: Optional[List[str]] = None,
    llm: Optional[ChatOpenAI] = None,
    model: str = "gpt-4o",
) -> Tuple[ComplianceRiskAnalysis, dict]:
    """
    Run compliance & risk analysis for a single CallAnalysis instance.

    Args:
        call: The call to analyze.
        required_phrases: List of required phrases (exact strings).
        forbidden_phrases: List of forbidden phrases (exact strings).
        llm: Optional shared ChatOpenAI client.
        model: LLM model name.

    Returns:
        (compliance_model, raw_json_dict)
    """
    if required_phrases is None:
        required_phrases = []
    if forbidden_phrases is None:
        forbidden_phrases = []

    if llm is None:
        llm = get_compliance_llm(model=model)

    prompt = build_compliance_prompt()

    call_metadata_json = json.dumps(
        call.metadata.model_dump(mode="json"),
        ensure_ascii=False,
    )

    required_phrases_json = json.dumps(required_phrases, ensure_ascii=False)
    forbidden_phrases_json = json.dumps(forbidden_phrases, ensure_ascii=False)

    chain = prompt | llm

    response = chain.invoke(
        {
            "call_metadata_json": call_metadata_json,
            "transcript": call.transcript,
            "required_phrases_json": required_phrases_json,
            "forbidden_phrases_json": forbidden_phrases_json,
        }
    )

    content = response.content
    if not isinstance(content, str):
        raise ValueError(f"Expected string content from LLM, got: {type(content)}")

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM JSON response: {e}\nRaw content: {content}") from e

    compliance = ComplianceRiskAnalysis.model_validate(data)

    return compliance, data
