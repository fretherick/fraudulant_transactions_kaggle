"""Download the public Kaggle dataset into the local data directory."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import kagglehub

DATASET_HANDLE = "chitwanmanchanda/fraudulent-transactions-data"


def download(destination: Path, force: bool = False) -> Path:
    destination = destination.resolve()
    if destination.exists() and not force:
        print(f"Dataset already exists: {destination}")
        return destination

    cache_directory = Path(kagglehub.dataset_download(DATASET_HANDLE))
    candidates = list(cache_directory.rglob("Fraud.csv"))
    if not candidates:
        raise FileNotFoundError(f"Fraud.csv was not found under {cache_directory}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidates[0], destination)
    print(f"Dataset copied to: {destination}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/Fraud.csv"))
    parser.add_argument("--force", action="store_true", help="Replace an existing CSV.")
    args = parser.parse_args()
    download(args.output, args.force)


if __name__ == "__main__":
    main()
