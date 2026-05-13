"""
Query a CORD-19 vector database and print the most relevant articles.

The database backend (pkl / faiss) is detected automatically from the file extension.
The metadata file can be given explicitly or discovered automatically from --data-dir
and --release (mirrors the interface of build_database.py).

Usage:
    # explicit metadata path (printed by build_database.py on completion)
    python query_database.py \\
        --db-file       db/vector_db.pkl \\
        --metadata-file data/2020-03-13/all_sources_metadata_2020-03-13.csv \\
        --query         "hypertension"

    # auto-discover metadata from data directory + release date
    python query_database.py \\
        --db-file    db/vector_db.pkl \\
        --data-dir   data/ \\
        --release    2020-03-13 \\
        --query      "hypertension"

    # custom model, FAISS backend, 20 results
    python query_database.py \\
        --db-file    db/vector_db.faiss \\
        --data-dir   data/ --release 2020-03-13 \\
        --query      "diabetes and COVID-19 susceptibility" \\
        --model      "sentence-transformers/all-MiniLM-L6-v2" \\
        --top-k      20
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_RELEASE = "2022-06-02"
SPECTER_MODEL_ID = "allenai/specter"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query a CORD-19 vector database for relevant articles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--db-file", required=True,
        help="Path to the saved database (vector_db.pkl or vector_db.faiss)",
    )

    # ── metadata: explicit path OR auto-discovery via data-dir + release ──────
    meta_group = parser.add_mutually_exclusive_group(required=True)
    meta_group.add_argument(
        "--metadata-file",
        help="Path to the metadata CSV (e.g. data/2020-03-13/all_sources_metadata_2020-03-13.csv)",
    )
    meta_group.add_argument(
        "--data-dir",
        help="Root data directory; metadata is discovered automatically (requires --release)",
    )

    parser.add_argument(
        "--release", default=DEFAULT_RELEASE,
        help="Release date (YYYY-MM-DD); used with --data-dir to locate metadata",
    )
    parser.add_argument(
        "--query", required=True,
        help='Query string, e.g. "hypertension"',
    )
    parser.add_argument(
        "--model", default=SPECTER_MODEL_ID,
        help="HuggingFace model ID (or local path) used to embed the query",
    )
    parser.add_argument(
        "--top-k", type=int, default=10,
        help="Number of results to return",
    )
    args = parser.parse_args()

    from src.cord19 import find_metadata_csv, find_release_dir, load_cord19_metadata, lookup_article
    from src.embedder import embed_query, load_model
    from src.vector_db import load_database

    db_path = Path(args.db_file)
    if not db_path.exists():
        print(f"Error: database not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    # Resolve metadata path
    if args.metadata_file:
        metadata_path = Path(args.metadata_file)
        if not metadata_path.exists():
            print(f"Error: metadata file not found: {metadata_path}", file=sys.stderr)
            sys.exit(1)
    else:
        release_dir = find_release_dir(Path(args.data_dir), args.release)
        try:
            metadata_path = find_metadata_csv(release_dir)
        except FileNotFoundError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)

    print(f"Loading database from {db_path} ...")
    vector_db = load_database(db_path)
    print(f"  {len(vector_db):,} vectors loaded")

    print(f"Loading model '{args.model}' ...")
    tokenizer, model = load_model(args.model)

    print(f"Embedding query: \"{args.query}\" ...")
    query_embedding = embed_query(args.query, tokenizer, model)

    print(f"Searching top {args.top_k} results ...")
    search_results = vector_db.search(query_embedding, top_k=args.top_k)

    metadata_df = load_cord19_metadata(str(metadata_path))

    print(f"\nTop {len(search_results)} articles for query: \"{args.query}\"")
    print("─" * 60)
    for rank, (key, score) in enumerate(search_results, start=1):
        article = lookup_article(metadata_df, key)
        title = article.get("title") or "N/A"
        doi = article.get("doi") or "N/A"
        print(f"{rank:2}. [{score:.4f}]  {title}")
        print(f"      doi: {doi}")
        print()


if __name__ == "__main__":
    main()
