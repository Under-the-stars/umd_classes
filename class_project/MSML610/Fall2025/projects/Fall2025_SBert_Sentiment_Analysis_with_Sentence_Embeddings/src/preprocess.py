# src/preprocess.py
import sys
import yaml
import pandas as pd
from pathlib import Path


def read_cfg():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def find_single_csv(raw_dir: Path) -> Path:
    csvs = list(raw_dir.glob("*.csv"))
    if len(csvs) == 0:
        raise FileNotFoundError(f"No CSV found in {raw_dir}. Place your dataset CSV there.")
    if len(csvs) > 1:
        raise RuntimeError(f"Multiple CSVs found in {raw_dir}. Keep only one CSV in {raw_dir}.")
    return csvs[0]


def read_csv_robust(path: Path) -> pd.DataFrame:
    # Try utf-8 (no header), then latin1 (no header).
    try:
        return pd.read_csv(path, encoding="utf-8", header=None)
    except UnicodeDecodeError:
        pass
    try:
        return pd.read_csv(path, encoding="latin1", header=None)
    except Exception:
        pass
    # As a last resort, try delimiters with latin1.
    try:
        return pd.read_csv(path, encoding="latin1", sep=";", header=None)
    except Exception:
        return pd.read_csv(path, encoding="latin1", sep=",", header=None)


def maybe_split_single_column(df: pd.DataFrame) -> pd.DataFrame:
    if df.shape[1] == 1:
        s = df.iloc[:, 0].astype(str)
        # Heuristic: if at least 20% of rows contain ';', use it; else comma.
        if (s.str.contains(";").mean() > 0.2):
            df = s.str.split(";", expand=True)
        else:
            df = s.str.split(",", expand=True)
    return df


def coerce_columns(df: pd.DataFrame, text_name: str, label_name: str, label_map: dict) -> pd.DataFrame:
    """
    Try to infer [label, text] vs [text, label] when headerless.
    Then rename to (text_name, label_name). Fall back to common names if present.
    """
    # If we have at least 2 columns, infer layout.
    if df.shape[1] >= 2:
        sample = df.head(100).copy()
        col0 = sample.iloc[:, 0].astype(str).str.strip().str.lower()
        col1 = sample.iloc[:, 1].astype(str)

        first_is_labelish = col0.isin(label_map.keys()).mean() > 0.6
        second_is_long_text = col1.str.len().mean() > 20

        if first_is_labelish and second_is_long_text:
            df = df.rename(columns={0: label_name, 1: text_name})
        else:
            # Try the reverse common case.
            df = df.rename(columns={0: text_name, 1: label_name})

    # Attempt fallback mapping from common header names if they already exist.
    rename_map = {}
    lower_map = {c.lower(): c for c in df.columns}

    for cand in ["sentence", "sentences", "text", "headline", "news", "content"]:
        if cand in lower_map:
            rename_map[lower_map[cand]] = text_name
            break

    for cand in ["sentiment", "label", "polarity", "class"]:
        if cand in lower_map:
            rename_map[lower_map[cand]] = label_name
            break

    df = df.rename(columns=rename_map)

    if text_name not in df.columns or label_name not in df.columns:
        raise ValueError(
            f"Could not find required columns. Got columns={list(df.columns)}. "
            f"Expected text='{text_name}', label='{label_name}'."
        )
    return df


def normalize_labels(df: pd.DataFrame, label_col: str, label_map: dict) -> pd.DataFrame:
    # Clean rows
    df = df.dropna(subset=[label_col]).copy()

    # If labels are strings, map them; else coerce to numeric
    if df[label_col].dtype == object:
        df[label_col] = (
            df[label_col]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(label_map)
        )
    else:
        df[label_col] = pd.to_numeric(df[label_col], errors="coerce")

    if df[label_col].isna().any():
        uniques = sorted(df[label_col].dropna().unique().tolist())
        raise ValueError(
            "Some labels could not be mapped to {0,1,2}. "
            f"Unique non-null labels seen: {uniques}. "
            "Update 'label_map' in config.yaml if your label strings differ."
        )

    df[label_col] = df[label_col].astype(int)
    return df


def main():
    cfg = read_cfg()

    raw_dir = Path(cfg["dataset"]["local_raw_dir"])
    out_csv = Path(cfg["outputs"]["processed_csv"])
    text_col = cfg["processing"]["text_column"]       # e.g., 'sentence'
    label_col = cfg["processing"]["label_column"]     # e.g., 'sentiment'
    label_map = {k.lower(): v for k, v in cfg["processing"]["label_map"].items()}

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    in_path = find_single_csv(raw_dir)
    df = read_csv_robust(in_path)
    df = maybe_split_single_column(df)
    df = coerce_columns(df, text_col, label_col, label_map)

    # Drop NA, dedup on text, normalize labels
    df = df.dropna(subset=[text_col]).drop_duplicates(subset=[text_col]).copy()
    df = normalize_labels(df, label_col, label_map)

    # Save + report
    df.to_csv(out_csv, index=False)
    counts = df[label_col].value_counts().sort_index().to_dict()
    print(
        f"[preprocess] OK -> {out_csv} "
        f"| rows={len(df)} | label_counts={counts} | cols={list(df.columns)}"
    )


if __name__ == "__main__":
    main()