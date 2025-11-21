from __future__ import annotations

import json
from typing import Optional, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from app.schemas import (
    CallAnalysis,
    SentimentAnalysis,
)


# Shared system prompt (strict JSON)
SYSTEM_PROMPT = """
You are an AI assistant that analyzes customer support or sales calls.

CRITICAL INSTRUCTIONS:
- You ALWAYS respond with a single valid JSON object.
- Do NOT include any explanations, markdown, backticks, or surrounding text.
- Do NOT include trailing commas.
- If you are unsure about a value, use null or an empty list [] as appropriate.
- Follow exactly the field names, types, and allowed values described in the user message.
""".strip()


# User prompt template for sentiment & emotion analysis
HUMAN_PROMPT = """
You are performing SENTIMENT AND EMOTION ANALYSIS for a single customer call.

You must respond with a SINGLE valid JSON object with exactly these fields:

- overall: one of "positive", "neutral", or "negative"
- score: a number between -1.0 and 1.0, or null
- emotion_tags: an array of strings
- sentiment_timeline: an array (possibly empty) of objects, each with:
    - segment_label: string
    - start_second: number or null
    - end_second: number or null
    - sentiment: one of "positive", "neutral", or "negative"
    - notes: string or null
- notes: string or null

Guidelines:
- "overall" reflects the entire call from the customer's perspective.
- "score": use approximately -1.0 (very negative) to +1.0 (very positive).
- "emotion_tags": 1-5 high-level emotions like "frustration", "confusion", "relief", "satisfaction", etc.
- "sentiment_timeline": 0-5 coarse segments that show sentiment changes. If you cannot infer timing, use null for start_second and end_second but still describe logical segments (e.g. "intro", "problem_explained", "resolution").
- "notes": 1-3 sentences summarizing the emotional journey.

Remember:
- Do NOT add any extra top-level fields.
- Do NOT include explanations or markdown.
- The output must be a single JSON object.

Call metadata (JSON):
{call_metadata_json}

Transcript:
\"\"\"{transcript}\"\"\"
""".strip()



def get_sentiment_llm(model: str = "gpt-4o", temperature: float = 0.0) -> ChatOpenAI:
    """
    Create a ChatOpenAI instance for sentiment analysis.
    Relies on OPENAI_API_KEY being set in the environment.
    """
    return ChatOpenAI(model=model, temperature=temperature)


def build_sentiment_prompt() -> ChatPromptTemplate:
    """
    Build the LangChain ChatPromptTemplate for sentiment analysis.
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ]
    )


def analyze_sentiment_for_call(
    call: CallAnalysis,
    llm: Optional[ChatOpenAI] = None,
    model: str = "gpt-4o",
) -> Tuple[SentimentAnalysis, dict]:
    """
    Run sentiment & emotion analysis for a single CallAnalysis instance.

    Returns:
        (sentiment_model, raw_json_dict)
    """
    if llm is None:
        llm = get_sentiment_llm(model=model)

    prompt = build_sentiment_prompt()

    call_metadata_json = json.dumps(
        call.metadata.model_dump(mode="json"),
        ensure_ascii=False,
    )

    # Build the chain: prompt -> llm
    chain = prompt | llm

    response = chain.invoke(
        {
            "call_metadata_json": call_metadata_json,
            "transcript": call.transcript,
        }
    )

    content = response.content
    # Ensure we have a string (llm might return structured content in future)
    if not isinstance(content, str):
        raise ValueError(f"Expected string content from LLM, got: {type(content)}")

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        # In a more advanced version, we could re-ask the model to fix JSON.
        raise ValueError(f"Failed to parse LLM JSON response: {e}\nRaw content: {content}") from e

    # Let Pydantic validate/normalize it into our SentimentAnalysis model
    sentiment = SentimentAnalysis.model_validate(data)

    return sentiment, data
