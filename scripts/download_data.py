"""
Download a CORD-19 release archive from the Semantic Scholar S3 bucket.

Each release is a single tar.gz at:
  https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/historical_releases/cord-19_<YYYY-MM-DD>.tar.gz

The archive contains:
  metadata.csv                     -- article metadata (titles, DOIs, abstracts …)
  cord_19_embeddings_<date>.csv    -- pre-computed SPECTER embeddings
  document_parses/                 -- full-text JSON parses (~100 GB, skipped by default)

Usage:
    python download_data.py --output-dir data/

    # also extract full-text parses
    python download_data.py --output-dir data/ --full-text

    # different release
    python download_data.py --output-dir data/ --release 2021-11-15

    # override the archive URL (e.g. mirror)
    python download_data.py --output-dir data/ --url https://example.com/cord-19_2022-06-02.tar.gz
"""

import argparse
import sys
import tarfile
import urllib.request
from pathlib import Path

CORD19_BASE = "https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com"
DEFAULT_RELEASE = "2022-06-02"


def _progress(block_count: int, block_size: int, total: int) -> None:
    downloaded = block_count * block_size
    if total > 0:
        pct = min(downloaded / total * 100, 100)
        print(f"\r  {pct:5.1f}%  {downloaded / 1e6:.1f} MB", end="", flush=True)


def download(url: str, dest: Path) -> None:
    print(f"Downloading {url}")
    try:
        urllib.request.urlretrieve(url, dest, reporthook=_progress)
    except Exception as exc:
        dest.unlink(missing_ok=True)
        print(f"\nFailed: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"\n  → {dest}  ({dest.stat().st_size / 1e6:.1f} MB)")


def extract(archive: Path, dest_dir: Path, include_full_text: bool) -> None:
    print(f"Extracting {archive.name} ...")
    with tarfile.open(archive) as tar:
        members = [
            m for m in tar.getmembers()
            if include_full_text or "document_parses" not in m.name
        ]
        tar.extractall(path=dest_dir, members=members)

    archive.unlink()

    for p in sorted(dest_dir.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(dest_dir)}  ({p.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and extract a CORD-19 release archive.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir", default="data",
        help="Directory to extract files into",
    )
    parser.add_argument(
        "--release", default=DEFAULT_RELEASE,
        help="Release date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--full-text", action="store_true",
        help="Also extract document_parses/ (~100 GB)",
    )
    parser.add_argument(
        "--url",
        help="Override the archive URL",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    archive_name = f"cord-19_{args.release}.tar.gz"
    url = args.url or f"{CORD19_BASE}/historical_releases/{archive_name}"
    archive_path = output_dir / archive_name

    # Skip if key output already exists
    embeddings_csv = next(output_dir.glob(f"cord_19_embeddings*{args.release}*.csv"), None)
    metadata_csv = output_dir / "metadata.csv"
    if embeddings_csv and metadata_csv.exists():
        print(f"Data for release {args.release} already present in {output_dir}/ — skipping download.")
        return

    download(url, archive_path)
    extract(archive_path, output_dir, include_full_text=args.full_text)


if __name__ == "__main__":
    main()
