"""
loader.py
---------
Data loading utilities for the DUC 2004 simulated dataset.

Provides functions to:
  - Load JSON dataset from disk
  - Validate dataset schema
  - Extract reference/system summary pairs as lists

Usage:
    from src.loader import load_dataset, get_pairs
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def load_dataset(path: str = "data/duc_2004_simulated.json") -> list[dict]:
    """
    Load the DUC 2004 simulated dataset from a JSON file.

    Args:
        path: Relative or absolute path to the JSON dataset file.

    Returns:
        List of sample dicts with keys: id, reference, system.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If the dataset structure is invalid.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{file_path}'. "
            "Run 'python data/duc_simulate.py' to generate it."
        )

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    _validate_dataset(data)
    logger.info("Loaded %d samples from '%s'.", len(data), file_path)
    return data


def _validate_dataset(data: list) -> None:
    """
    Validate that each sample contains required fields.

    Args:
        data: Raw list loaded from JSON.

    Raises:
        ValueError: On schema violations.
    """
    required_keys = {"id", "reference", "system"}
    for i, sample in enumerate(data):
        missing = required_keys - set(sample.keys())
        if missing:
            raise ValueError(
                f"Sample at index {i} is missing required keys: {missing}"
            )
        for key in ("reference", "system"):
            if not isinstance(sample[key], str) or not sample[key].strip():
                raise ValueError(
                    f"Sample '{sample.get('id', i)}' has empty or invalid '{key}'."
                )


def get_pairs(
    data: list[dict],
    limit: Optional[int] = None,
) -> tuple[list[str], list[str], list[str]]:
    """
    Extract IDs, reference summaries, and system summaries from dataset.

    Args:
        data: Dataset as returned by load_dataset().
        limit: Optional cap on number of samples to use.

    Returns:
        Tuple of (ids, references, systems).
    """
    if limit is not None:
        data = data[:limit]

    ids = [s["id"] for s in data]
    references = [s["reference"] for s in data]
    systems = [s["system"] for s in data]

    return ids, references, systems


def describe_dataset(data: list[dict]) -> None:
    """
    Print a human-readable summary of the dataset.

    Args:
        data: Dataset as returned by load_dataset().
    """
    print(f"\n{'='*60}")
    print(f"  DUC 2004 Dataset Summary")
    print(f"{'='*60}")
    print(f"  Total samples   : {len(data)}")
    ref_lens = [len(s["reference"].split()) for s in data]
    sys_lens = [len(s["system"].split()) for s in data]
    print(f"  Avg ref length  : {sum(ref_lens)/len(ref_lens):.1f} words")
    print(f"  Avg sys length  : {sum(sys_lens)/len(sys_lens):.1f} words")
    print(f"  Sample IDs      : {[s['id'] for s in data[:3]]} ...")
    print(f"{'='*60}\n")
