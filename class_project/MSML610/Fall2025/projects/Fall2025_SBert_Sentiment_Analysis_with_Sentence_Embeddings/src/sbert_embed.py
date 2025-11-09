import sys, yaml, numpy as np, pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Load config
cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
cfg = yaml.safe_load(open(cfg_path))

csv = Path(cfg["outputs"]["processed_csv"])
emb_out = Path(cfg["outputs"]["embeddings_npy"])
lab_out = Path(cfg["outputs"]["labels_npy"])

text_col = cfg["processing"]["text_column"]       # 'sentence'
label_col = cfg["processing"]["label_column"]     # 'sentiment'

# Read cleaned data
df = pd.read_csv(csv)

# Load model (MiniLM is fast & small)
model_name = cfg["embeddings"]["model"]
batch_size = cfg["embeddings"]["batch_size"]
model = SentenceTransformer(model_name)

# Encode to numpy
emb = model.encode(
    df[text_col].astype(str).tolist(),
    batch_size=batch_size,
    convert_to_numpy=True,
    show_progress_bar=True,
)

# Save outputs
emb_out.parent.mkdir(parents=True, exist_ok=True)
np.save(emb_out, emb)
np.save(lab_out, df[label_col].values.astype(int))

print(f"[embed] Saved: {emb_out} and {lab_out} | shape={emb.shape}")