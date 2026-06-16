"""
inspect_datasets.py — Run this FIRST whenever a dataset gives 0 samples.

It loads each dataset, prints exact column names, and shows a sample row
so you can see which column contains the judgment text.

Usage:
    python finetuning/inspect_datasets.py
"""

from datasets import load_dataset
from loguru import logger

DATASETS_TO_CHECK = [
    # (dataset_id, config_or_None, split)
    ("Exploration-Lab/IL-TUR",                            "bail",  "train"),
    ("Exploration-Lab/IL-TUR",                            "cjpe",  "train"),
    ("Exploration-Lab/IL-TUR",                            "summ",  "train"),
    ("opennyaiorg/InJudgements_dataset",                  None,    "train"),
    ("rishiai/indian-court-judgements-and-its-summaries", None,    "train"),
    ("nisaar/Lawyer_GPT_India",                           None,    "train"),
]


def inspect(dataset_id: str, config: str | None, split: str):
    name = f"{dataset_id}" + (f"/{config}" if config else "")
    print(f"\n{'='*60}")
    print(f"Dataset: {name}")
    print(f"{'='*60}")

    try:
        if config:
            ds = load_dataset(dataset_id, config, split=split, trust_remote_code=False)
        else:
            ds = load_dataset(dataset_id, split=split, trust_remote_code=False)

        print(f"✓ Loaded: {len(ds)} rows")
        print(f"  Columns: {ds.column_names}")
        print(f"\n  First row:")

        row = ds[0]
        for k, v in row.items():
            if v is None:
                print(f"    [{k}]: None")
            elif isinstance(v, str):
                preview = v[:200].replace("\n", " ")
                print(f"    [{k}] (len={len(v)}): {preview}")
            elif isinstance(v, list):
                print(f"    [{k}] (list, len={len(v)}): {str(v)[:100]}")
            else:
                print(f"    [{k}]: {str(v)[:100]}")

        # Tell user which column to use
        text_cols = [k for k, v in row.items() if isinstance(v, str) and len(v) > 500]
        if text_cols:
            best = max(text_cols, key=lambda k: len(row[k]))
            print(f"\n  ✅ Best text column: '{best}' (len={len(row[best])})")
            print(f"     Add to _extract_text(): row.get('{best}')")
        else:
            print(f"\n  ⚠ No column with >500 chars found. All string columns:")
            for k, v in row.items():
                if isinstance(v, str):
                    print(f"     [{k}] len={len(v)}: {v[:80]}")

    except Exception as e:
        print(f"✗ Failed: {e}")


if __name__ == "__main__":
    for dataset_id, config, split in DATASETS_TO_CHECK:
        inspect(dataset_id, config, split)

    print(f"\n{'='*60}")
    print("Once you know the correct column names, update _extract_text()")
    print("in finetuning/prepare_dataset.py and re-run prepare_dataset.py")
    print(f"{'='*60}")
