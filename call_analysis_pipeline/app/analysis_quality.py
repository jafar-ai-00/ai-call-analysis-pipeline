from __future__ import annotations

import json
from typing import Optional, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from app.schemas import (
    CallAnalysis,
    CallQualityAnalysis,
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
You are performing CALL QUALITY AND AGENT PERFORMANCE ANALYSIS for a single customer call.

You must respond with a SINGLE valid JSON object with exactly these fields:

- overall_quality_score: integer between 0 and 100, or null
- scores: an object with the following optional integer fields (0–100 or null):
    - greeting
    - listening_and_empathy
    - clarity_of_explanations
    - professionalism
    - script_adherence
- strengths: array of strings
- improvements: array of strings
- notes: string or null

Guidelines:
- All scores should be between 0 and 100 where:
  - 0–39: poor
  - 40–59: fair
  - 60–79: good
  - 80–100: excellent
- "overall_quality_score" is NOT a simple average; adjust based on your judgment of the whole call.
- "strengths": 2–5 concrete points about what the agent did well (behavior-level, not generic praise).
- "improvements": 2–5 specific, actionable suggestions (what to do differently next time).
- "notes": optional narrative summary of quality (1–3 sentences).

Remember:
- Do NOT add extra top-level fields.
- Do NOT include explanations or markdown.
- The output must be a single JSON object.

Call metadata (JSON):
{call_metadata_json}

Transcript:
\"\"\"{transcript}\"\"\"
""".strip()


def get_quality_llm(model: str = "gpt-4o", temperature: float = 0.0) -> ChatOpenAI:
    """
    Create a ChatOpenAI instance for call quality analysis.
    """
    return ChatOpenAI(model=model, temperature=temperature)


def build_quality_prompt() -> ChatPromptTemplate:
    """
    Build the LangChain ChatPromptTemplate for call quality analysis.
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ]
    )


def analyze_quality_for_call(
    call: CallAnalysis,
    llm: Optional[ChatOpenAI] = None,
    model: str = "gpt-4o",
) -> Tuple[CallQualityAnalysis, dict]:
    """
    Run call quality & agent performance analysis for a single CallAnalysis instance.

    Returns:
        (quality_model, raw_json_dict)
    """
    if llm is None:
        llm = get_quality_llm(model=model)

    prompt = build_quality_prompt()

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

    quality = CallQualityAnalysis.model_validate(data)

    return quality, data
