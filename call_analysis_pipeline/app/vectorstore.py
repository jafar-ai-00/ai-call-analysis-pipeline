import os
import json
from pathlib import Path
from typing import List, Dict, Any

import chromadb
import yaml
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from app.schemas import CallAnalysis



CONFIG_PATH = Path("config/config.yaml")


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found at {CONFIG_PATH}")
    with CONFIG_PATH.open("r") as f:
        return yaml.safe_load(f)


class OpenAIEmbedder:
    """
    Simple embedding function for Chroma that uses OpenAI Embeddings.

    Chroma expects an object with:
      - __call__(input: List[str]) -> List[List[float]]
      - embed_documents(input: List[str]) -> List[List[float]]
      - embed_query(input: str) -> List[float]
      - name() -> str
      - (optionally) to_dict() -> dict
    """

    def __init__(self, model: str = "text-embedding-3-small"):
        self.client = OpenAI()
        self.model = model

    def _embed_batch(self, texts) -> List[List[float]]:
        """Internal helper to call OpenAI once for a batch of texts.
        Normalizes whatever Chroma passes into a clean List[str].
        """
        if not texts:
            return []

        # Normalize to list of strings
        # Chroma *should* pass a list of strings, but we’re defensive.
        if isinstance(texts, str):
            input_texts = [texts]
        else:
            try:
                # If it's iterable, turn each element into a string
                _ = iter(texts)
                input_texts = [str(t) for t in texts]
            except TypeError:
                # Not iterable, just make it a single-element list
                input_texts = [str(texts)]

        resp = self.client.embeddings.create(model=self.model, input=input_texts)
        return [item.embedding for item in resp.data]

    # Chroma 0.4+ may still call __call__ directly
    def __call__(self, input) -> List[List[float]]:
        return self._embed_batch(input)

    def embed_documents(self, input) -> List[List[float]]:
        return self._embed_batch(input)

    def embed_query(self, input: str) -> List[float]:
        return self._embed_batch([input])[0]

    def name(self) -> str:
        """A short name that Chroma uses to detect embedding function configuration."""
        return f"openai-embeddings-{self.model}"

    def to_dict(self) -> Dict[str, Any]:
        """Configuration payload so Chroma can persist and compare."""
        return {
            "provider": "openai",
            "model": self.model,
        }



def get_chroma_client(persist_directory: str | Path) -> chromadb.PersistentClient:
    """
    Create (or connect to) a persistent Chroma client.
    """
    return chromadb.PersistentClient(path=str(persist_directory))


def get_calls_collection(
    client: chromadb.PersistentClient,
    embedder: OpenAIEmbeddingFunction,
    collection_name: str = "calls",
):
    """
    Get or create the Chroma collection used for call transcripts.
    """
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedder,
    )


def list_call_json_files(data_dir: str | Path = "data") -> List[Path]:
    """
    List all call JSON files under data/calls.
    """
    base = Path(data_dir)
    calls_dir = base / "calls"
    if not calls_dir.exists():
        return []
    return sorted(calls_dir.glob("*.json"))


def load_call_from_json(path: Path) -> CallAnalysis:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return CallAnalysis.model_validate(data)


def build_calls_index(
    data_dir: str | Path = "data",
    chroma_db_dir: str | Path = "chroma_db",
    collection_name: str = "calls",
) -> None:
    ...
    config = load_config()
    openai_conf = config.get("openai", {}) or {}
    embedding_model = openai_conf.get("embedding_model", "text-embedding-3-small")
    openai_env_var = openai_conf.get("api_key_env", "OPENAI_API_KEY")
    api_key = os.getenv(openai_env_var)

    data_dir = Path(data_dir)
    chroma_db_dir = Path(chroma_db_dir)

    call_paths = list_call_json_files(data_dir=data_dir)
    if not call_paths:
        print("No call JSON files found in data/calls. Run the pipeline first.")
        return

    print(f"Building Chroma index from {len(call_paths)} call file(s)...")

    # ✅ Use Chroma's built-in OpenAI embedding function
    embedder = OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name=embedding_model,
    )
    client = get_chroma_client(persist_directory=chroma_db_dir)
    collection = get_calls_collection(client, embedder, collection_name=collection_name)

    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for path in call_paths:
        call = load_call_from_json(path)
        call_id = call.metadata.call_id
        transcript = call.transcript or ""
        if not transcript.strip():
            continue

        sentiment_label = (
            call.sentiment.overall.value if call.sentiment and call.sentiment.overall else None
        )
        primary_intent = (
            call.intent_and_topics.primary_intent
            if call.intent_and_topics and call.intent_and_topics.primary_intent
            else None
        )
        risk_label = (
            call.compliance_and_risk.risk_level.value
            if call.compliance_and_risk and call.compliance_and_risk.risk_level
            else None
        )
        quality_score = (
            call.call_quality.overall_quality_score
            if call.call_quality and call.call_quality.overall_quality_score is not None
            else None
        )

        ids.append(call_id)
        documents.append(transcript)
        metadatas.append(
            {
                "call_id": call_id,
                "client_id": call.metadata.client_id,
                "audio_file": call.metadata.audio_file,
                "sentiment": sentiment_label,
                "primary_intent": primary_intent,
                "risk_level": risk_label,
                "quality_score": quality_score,
            }
        )

    if not ids:
        print("No non-empty transcripts found to index.")
        return

    print(f"Upserting {len(ids)} document(s) into Chroma collection '{collection_name}'...")
    # Upsert so we can rebuild without errors on duplicate IDs
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
    )
    print("Index build complete ✅")


def semantic_search_calls(
    query: str,
    n_results: int = 5,
    chroma_db_dir: str | Path = "chroma_db",
    collection_name: str = "calls",
) -> List[Dict[str, Any]]:
    """
    Perform semantic search over indexed calls.

    Returns:
        List of matches, each containing:
        - id
        - document (transcript chunk)
        - metadata (call_id, client_id, etc.)
        - distance (smaller = closer)
    """
    config = load_config()
    openai_conf = config.get("openai", {}) or {}
    embedding_model = openai_conf.get("embedding_model", "text-embedding-3-small")
    openai_env_var = openai_conf.get("api_key_env", "OPENAI_API_KEY")
    api_key = os.getenv(openai_env_var)

    chroma_db_dir = Path(chroma_db_dir)

    # ✅ Use Chroma's built-in embedding function again
    embedder = OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name=embedding_model,
    )
    client = get_chroma_client(persist_directory=chroma_db_dir)
    collection = get_calls_collection(client, embedder, collection_name=collection_name)

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
    )

    # Chroma returns lists-of-lists
    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    matches: List[Dict[str, Any]] = []
    for i in range(len(ids)):
        matches.append(
            {
                "id": ids[i],
                "document": docs[i],
                "metadata": metas[i],
                "distance": distances[i],
            }
        )

    return matches
