from app.vectorstore import build_calls_index
from pathlib import Path
import yaml


CONFIG_PATH = Path("config/config.yaml")


def load_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found at {CONFIG_PATH}")
    with CONFIG_PATH.open("r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    data_dir = config.get("data_dir", "data/")
    chroma_db_dir = config.get("chroma_db_dir", "chroma_db/")
    print("=== Building Chroma index over calls ===")
    build_calls_index(data_dir=data_dir, chroma_db_dir=chroma_db_dir, collection_name="calls")


if __name__ == "__main__":
    main()
