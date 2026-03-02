"""Dataset ingestion module for the BIRD-Interact evaluation framework.

Provides functionality to download and cache the BIRD-Interact dataset
from HuggingFace, using local Parquet files for efficient repeated access.
"""

from pathlib import Path

import pandas as pd
from datasets import load_dataset

# Default data directory at project root (outside src/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_DATA_DIR = _PROJECT_ROOT / "data"


def _parquet_path(data_dir: Path, hf_dataset_name: str, split: str) -> Path:
    """Build a deterministic Parquet file path from dataset name and split.

    Args:
        data_dir: Directory where Parquet files are stored.
        hf_dataset_name: HuggingFace dataset identifier.
        split: Dataset split name.

    Returns:
        Path to the Parquet file.
    """
    safe_name = hf_dataset_name.replace("/", "__")
    return data_dir / f"{safe_name}__{split}.parquet"


def load_or_download_dataset(
    hf_dataset_name: str,
    split: str = "train",
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """Load a HuggingFace dataset, using a local Parquet cache when available.

    If a cached ``.parquet`` file already exists in *data_dir*, it is loaded
    directly.  Otherwise the dataset is downloaded via the ``datasets``
    library, saved as Parquet, and then returned as a :class:`pandas.DataFrame`.

    Args:
        hf_dataset_name: HuggingFace dataset identifier
            (e.g. ``"bird-bench/BIRD-interact-mini"``).
        split: Dataset split to load (default ``"train"``).
        data_dir: Directory to store/load cached Parquet files.
            Defaults to ``<project_root>/data/``.

    Returns:
        A :class:`pandas.DataFrame` with the dataset contents.
    """
    if data_dir is None:
        data_dir = _DEFAULT_DATA_DIR

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    parquet_file = _parquet_path(data_dir, hf_dataset_name, split)

    if parquet_file.exists():
        return pd.read_parquet(parquet_file)

    ds = load_dataset(hf_dataset_name, split=split)
    df: pd.DataFrame = ds.to_pandas()  # type: ignore[union-attr]
    df.to_parquet(parquet_file, index=False)
    return df
