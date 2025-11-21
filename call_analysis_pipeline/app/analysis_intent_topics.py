from __future__ import annotations

import json
from typing import Optional, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from app.schemas import (
    CallAnalysis,
    IntentTopicsAnalysis,
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
You are performing INTENT AND TOPICS ANALYSIS for a single customer call.

You must respond with a SINGLE valid JSON object with exactly these fields:

- primary_intent: string or null
- secondary_intents: array of strings
- topics: array of strings
- key_phrases: array of strings
- intent_confidence: number between 0.0 and 1.0, or null
- notes: string or null

Guidelines:
- "primary_intent": the main reason for the call, e.g. "book_appointment", "reschedule_appointment",
  "billing_question", "complaint", "technical_issue", "general_inquiry".
- "secondary_intents": other relevant intents, if any.
- "topics": broader themes, e.g. "scheduling", "pricing", "refunds", "product_information",
  "support", "onboarding".
- "key_phrases": 3–10 short important phrases the customer or agent said that relate to the main intents
  (paraphrasing is OK).
- "intent_confidence": between 0.0 and 1.0 based on how clear the primary intent is.
- "notes": 1–3 sentences explaining your reasoning if the intent is ambiguous.

Remember:
- Do NOT add extra top-level fields.
- Do NOT include explanations or markdown.
- The output must be a single JSON object.

Call metadata (JSON):
{call_metadata_json}

Transcript:
\"\"\"{transcript}\"\"\"
""".strip()


def get_intent_llm(model: str = "gpt-4o", temperature: float = 0.0) -> ChatOpenAI:
    """
    Create a ChatOpenAI instance for intent & topics analysis.
    """
    return ChatOpenAI(model=model, temperature=temperature)


def build_intent_prompt() -> ChatPromptTemplate:
    """
    Build the LangChain ChatPromptTemplate for intent & topics analysis.
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ]
    )


def analyze_intent_topics_for_call(
    call: CallAnalysis,
    llm: Optional[ChatOpenAI] = None,
    model: str = "gpt-4o",
) -> Tuple[IntentTopicsAnalysis, dict]:
    """
    Run intent & topics analysis for a single CallAnalysis instance.

    Returns:
        (intent_topics_model, raw_json_dict)
    """
    if llm is None:
        llm = get_intent_llm(model=model)

    prompt = build_intent_prompt()

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

    intent_topics = IntentTopicsAnalysis.model_validate(data)

    return intent_topics, data
