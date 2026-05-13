from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np
import pandas as pd


# ── release/file discovery ────────────────────────────────────────────────────

def find_release_dir(data_dir: Path, release: str) -> Path:
    """Return the directory that contains the release files.

    download_data.py extracts archives into data/<release>/.
    Falls back to data_dir itself for flat layouts.
    """
    candidate = data_dir / release
    if candidate.is_dir():
        return candidate
    return data_dir


def find_metadata_csv(release_dir: Path) -> Path:
    """Return the metadata CSV, trying several known naming conventions."""
    for pattern in ["metadata.csv", "all_sources_metadata_*.csv", "*metadata*.csv"]:
        matches = sorted(release_dir.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"No metadata CSV found in {release_dir}. "
        f"Expected metadata.csv or all_sources_metadata_<date>.csv."
    )


# ── embeddings loading ────────────────────────────────────────────────────────

def iter_cord19_embeddings(filepath: str) -> Iterator[Tuple[str, np.ndarray]]:
    """Yield (cord_uid, embedding) pairs from a CORD-19 embeddings CSV."""
    df = pd.read_csv(filepath, header=None)
    for _, row in df.iterrows():
        cord_uid = row[0]
        embedding = np.asarray(row[1:].tolist())
        yield cord_uid, embedding


def load_cord19_metadata(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath, low_memory=False)


def _key_column(metadata_df: pd.DataFrame) -> str:
    """Return the article identifier column: cord_uid (newer) or sha (older releases)."""
    for col in ("cord_uid", "sha"):
        if col in metadata_df.columns:
            return col
    raise ValueError(
        f"No article identifier column found. Expected 'cord_uid' or 'sha'. "
        f"Got: {list(metadata_df.columns)}"
    )


def lookup_article(metadata_df: pd.DataFrame, key: str) -> dict:
    """Return title, doi, and abstract for a given article key.

    Works with both newer releases (cord_uid) and older releases (sha).
    Returns {} when the key is not found.
    """
    key_col = _key_column(metadata_df)
    row = metadata_df[metadata_df[key_col].astype(str) == key]
    if row.empty:
        return {}
    return {
        key_col: key,
        "doi": row["doi"].iloc[0] if "doi" in row.columns else None,
        "title": row["title"].iloc[0] if "title" in row.columns else None,
        "abstract": row["abstract"].iloc[0] if "abstract" in row.columns else None,
    }
