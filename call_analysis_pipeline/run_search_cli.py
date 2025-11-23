import sys
from pathlib import Path
import yaml

from app.vectorstore import semantic_search_calls

CONFIG_PATH = Path("config/config.yaml")


def load_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found at {CONFIG_PATH}")
    with CONFIG_PATH.open("r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    chroma_db_dir = config.get("chroma_db_dir", "chroma_db/")

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Enter your search query: ").strip()

    if not query:
        print("No query provided.")
        return

    print(f"🔍 Searching for: {query!r}")
    matches = semantic_search_calls(
        query=query,
        n_results=5,
        chroma_db_dir=chroma_db_dir,
        collection_name="calls",
    )

    if not matches:
        print("No results.")
        return

    print(f"\nTop {len(matches)} result(s):\n")
    for i, m in enumerate(matches, start=1):
        meta = m["metadata"] or {}
        call_id = meta.get("call_id")
        client_id = meta.get("client_id")
        sentiment = meta.get("sentiment")
        primary_intent = meta.get("primary_intent")
        risk_level = meta.get("risk_level")
        quality_score = meta.get("quality_score")
        distance = m.get("distance")

        # Short preview of the matching text
        doc = (m["document"] or "")[:200].replace("\n", " ")
        if len(m["document"] or "") > 200:
            doc += "..."

        print(f"Result {i}:")
        print(f"  call_id       : {call_id}")
        print(f"  client_id     : {client_id}")
        print(f"  sentiment     : {sentiment}")
        print(f"  primary_intent: {primary_intent}")
        print(f"  risk_level    : {risk_level}")
        print(f"  quality_score : {quality_score}")
        print(f"  distance      : {distance}")
        print(f"  text          : {doc}")
        print("-" * 60)


if __name__ == "__main__":
    main()
