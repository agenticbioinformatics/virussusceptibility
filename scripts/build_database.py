"""
Build a CORD-19 vector database and save it.

The release archive (produced by download_data.py) extracts into a date-named
subdirectory, e.g. data/2020-03-13/.  This script locates the right files
automatically and supports two modes:

  Mode A  (pre-computed embeddings)
    Newer releases include cord_19_embeddings_<date>.csv with SPECTER vectors
    already computed.  The script loads those directly — fast, no GPU needed.

  Mode B  (embed from metadata)
    Older releases (e.g. 2020-03-13) ship only a metadata CSV.  The script
    embeds title + abstract for every article using a HuggingFace model.

In both modes the key stored in the database is the article identifier found
in the source file (cord_uid if present, otherwise sha).

Output: vector_db.pkl  (default)
     or vector_db.faiss + vector_db.faiss.keys.pkl  (--db-backend faiss)

Usage:
    # auto-detect mode, default release
    python build_database.py --data-dir data/ --output-dir db/ --release 2020-03-13

    # FAISS backend
    python build_database.py --data-dir data/ --output-dir db/ --release 2020-03-13 --db-backend faiss

    # limit articles embedded in Mode B (useful for testing)
    python build_database.py --data-dir data/ --output-dir db/ --release 2020-03-13 --max-articles 5000
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_RELEASE = "2022-06-02"
DEFAULT_MODEL = "allenai/specter"
LOG_INTERVAL_PRECOMPUTED = 50_000
LOG_INTERVAL_EMBED = 500


def find_embeddings_csv(release_dir: Path, release: str) -> Optional[Path]:
    """Return the pre-computed embeddings CSV if present, else None."""
    for pattern in [
        f"cord_19_embeddings_{release}.csv",
        "cord_19_embeddings.csv",
        "cord_19_embeddings*.csv",
    ]:
        matches = sorted(release_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


# ── metadata loading ──────────────────────────────────────────────────────────

def load_papers_from_metadata(metadata_path: Path, max_articles: Optional[int]):
    """Return (key_col, keys, papers) from a metadata CSV.

    Supports both cord_uid (newer releases) and sha (older releases) as key.
    Papers is a list of {"title": ..., "abstract": ...} dicts.
    """
    import pandas as pd

    df = pd.read_csv(metadata_path, low_memory=False)

    if "cord_uid" in df.columns:
        key_col = "cord_uid"
    elif "sha" in df.columns:
        key_col = "sha"
    else:
        raise ValueError(
            f"Cannot find an article identifier in {metadata_path.name}.\n"
            f"Expected a 'cord_uid' or 'sha' column. Found: {list(df.columns)}"
        )

    # Drop rows with no title and no key
    df = df.dropna(subset=[key_col, "title"])

    if max_articles is not None:
        df = df.head(max_articles)

    keys = df[key_col].astype(str).tolist()
    papers = [
        {"title": str(row["title"]), "abstract": str(row.get("abstract", ""))}
        for _, row in df.iterrows()
    ]
    return key_col, keys, papers


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a CORD-19 vector database.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", required=True,
        help="Root data directory (output of download_data.py)",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory to save the database files",
    )
    parser.add_argument(
        "--release", default=DEFAULT_RELEASE,
        help="Release date (YYYY-MM-DD) — used to locate the release subdirectory",
    )
    parser.add_argument(
        "--db-backend", choices=["pkl", "faiss"], default="pkl",
        help="Storage backend: pkl (single file) or faiss (requires faiss-cpu)",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help="HuggingFace model used in Mode B (embed from metadata)",
    )
    parser.add_argument(
        "--max-articles", type=int, default=None,
        help="Limit number of articles embedded in Mode B (default: all)",
    )
    args = parser.parse_args()

    from src.cord19 import find_metadata_csv, find_release_dir
    from src.vector_db import FaissVectorDatabase, VectorDatabase, save_database

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    release_dir = find_release_dir(data_dir, args.release)
    print(f"Release directory: {release_dir}")

    db = FaissVectorDatabase() if args.db_backend == "faiss" else VectorDatabase()

    embeddings_csv = find_embeddings_csv(release_dir, args.release)

    if embeddings_csv:
        # ── Mode A: load pre-computed embeddings ──────────────────────────────
        from src.cord19 import iter_cord19_embeddings

        print(f"Mode A: loading pre-computed embeddings from {embeddings_csv.name} ...")
        for i, (key, embedding) in enumerate(iter_cord19_embeddings(str(embeddings_csv))):
            db.insert(key, embedding)
            if (i + 1) % LOG_INTERVAL_PRECOMPUTED == 0:
                print(f"  {i + 1:,} vectors inserted ...")
    else:
        # ── Mode B: embed from metadata ───────────────────────────────────────
        from src.embedder import embed_paper, load_model

        metadata_path = find_metadata_csv(release_dir)
        print(f"Mode B: no pre-computed embeddings found.")
        print(f"  Metadata: {metadata_path.name}")
        print(f"  Model:    {args.model}")

        key_col, keys, papers = load_papers_from_metadata(metadata_path, args.max_articles)
        print(f"  Articles: {len(papers):,}  (key column: '{key_col}')")
        if args.max_articles is None and len(papers) > 10_000:
            print(f"  Warning: embedding {len(papers):,} articles will take a long time. "
                  f"Use --max-articles to limit.")

        print("  Loading model ...")
        tokenizer, model = load_model(args.model)

        for i, (key, paper) in enumerate(zip(keys, papers)):
            embedding = embed_paper(paper, tokenizer, model)
            db.insert(key, embedding)
            if (i + 1) % LOG_INTERVAL_EMBED == 0:
                print(f"  {i + 1:,} / {len(papers):,} articles embedded ...")

    print(f"Total vectors: {len(db):,}")

    suffix = ".faiss" if args.db_backend == "faiss" else ".pkl"
    db_file = output_dir / f"vector_db{suffix}"
    print(f"Saving to {db_file} ...")
    save_database(db, db_file)

    sizes = ", ".join(
        f"{p.stat().st_size / 1e6:.1f} MB"
        for p in sorted(output_dir.iterdir())
        if "vector_db" in p.name
    )
    print(f"Done.  ({sizes})")

    metadata_path = find_metadata_csv(release_dir)
    print(f"\nTo query this database run:")
    print(f"  python query_database.py \\")
    print(f"      --db-file       {db_file} \\")
    print(f"      --metadata-file {metadata_path} \\")
    print(f"      --query         \"hypertension\"")


if __name__ == "__main__":
    main()
