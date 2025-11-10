````markdown
# DGL API — Link Prediction for Movie Recommendations (MSML610)

## Overview
We demonstrate the **native DGL API** (heterographs, node/edge data, GraphSAGE) and a thin wrapper (`dgl_utils.py`) to build a **user–movie** bipartite graph with **ratings** as edge weights, learn embeddings, and evaluate **Precision@K / Recall@K** (implicit) and **RMSE** (explicit).

## Problem & Goals
- Treat recommendations as **link prediction** on a bipartite graph (users ↔ movies).
- Optimize for **top-N** recommendation quality; report **P@K / R@K**.
- Also regress ratings from learned embeddings → **RMSE**.

## Native DGL API we rely on (brief)
- `dgl.heterograph(...)` to create user→movie edges and reverse edges.  
- `g.edges[etype].data["rating"]` to store **edge weights**.  
- `dgl.to_homogeneous(g)` to use a simple homogeneous **GraphSAGE** encoder.  
- `dgl.nn.SAGEConv` for message passing.

## Our Thin Wrapper (`dgl_utils.py`)
- **Data**: `load_movielens`, `remap_ids`.  
- **Features**: `build_movie_genre_onehot` (movie genres → node features).  
- **Graph**: `build_bipartite_graph` (edge data includes ratings & timestamp).  
- **Splits**: `make_edge_splits`, `eids_to_pairs`.  
- **Model**: `GraphSAGEModel`, `LinkPredictorDot`.  
- **Train**: `train_link_prediction` (pos/neg sampling + BCE).  
- **Metrics**: `evaluate_precision_recall_at_k`, `fit_edge_regressor_ridge`, `rmse_from_regressor`.  
- **Recs**: `build_user_seen_map`, `recommend_topk_for_user`, `id_maps_to_title_lookup`.

> Design choice: keep the API **small and composable** so notebooks show *“Explain → Run → Inspect”* with one call per step.

## How to Run
- **Notebook**: `DGL.API.ipynb` (select the provided kernel).  
- **Script (optional)**:  
  ```bash
  python DGL.API.py --ratings data/ratings.csv --movies data/movies.csv \
                    --max-edges 100000 --epochs 2 --k 10
````

* If CSVs aren’t present, we fallback to a **toy** dataset so the tutorial always runs.

## Evaluation

* **Precision@K / Recall@K** on held-out edges using dot-product scores.
* **RMSE**: freeze embeddings, fit a **ridge** regressor on train edges, score on test edges.

## Design Decisions & Trade-offs

* Homogeneous GraphSAGE via `dgl.to_homogeneous` for clarity (teaching).
* Genre one-hot features fused with learnable movie embeddings (simple, fast).
* Small epochs and random negative sampling for the API demo; advanced sampling, neighbor loaders, and hetero encoders belong in the **Example** notebook.

## Limitations & Next Steps

* Add **val-based early stopping**, **harder negatives** (exclude known positives), and **heterogeneous GNNs** (e.g., type-specific layers).
* Add temporal splits and side info (directors/actors/tags) as extra node/edge types.

## References

* DGL documentation (GraphSAGE, heterographs).
* MovieLens 20M dataset (Harper & Konstan, 2015).

```

---

# 3) **`DGL.API.ipynb`** — template-compliant, cell-by-cell

> Keep each code cell **short** and call a single function from `dgl_utils.py`. Put the explanation just above each cell.

1) **Title (Markdown)**  
   “DGL API — Graph Construction, Node Features, GraphSAGE & Evaluation”

2) **Imports & Seed (Code)**  
   - `import torch, dgl, pandas as pd, numpy as np, logging`  
   - `import dgl_utils as du`  
   - set seeds, print versions.

3) **Config (Code)**  
   - Paths: `DATA_DIR`, `RATINGS_CSV`, `MOVIES_CSV`  
   - Small params: `MAX_EDGES=100000`, `EPOCHS=2`, `EMBED_DIM=32`, `K=10`, `SEED=42`.

4) **Load data (Markdown + Code)**  
   - Explain “real CSVs if present; else toy”.  
   - Call a local helper (or inline the same logic your script uses).

5) **Remap IDs (Markdown + Code)**  
   - Call `du.remap_ids(...)`; print counts of users/movies/edges.

6) **Build heterograph (Markdown + Code)**  
   - Call `du.build_bipartite_graph(...)`; print node/edge types and first few `rating` edge data.

7) **Node features (Markdown + Code)**  
   - If `movies.csv` exists → `du.build_movie_genre_onehot(...)` and print shape + #genres.

8) **Edge splits (Markdown + Code)**  
   - `du.make_edge_splits(...)` and `du.eids_to_pairs(...)`.  
   - Collect ratings per split for RMSE.

9) **Train GraphSAGE (Markdown + Code)**  
   - Call `du.train_link_prediction(...)` with small `epochs`; capture `user_emb`, `movie_emb`.

10) **P@K / R@K (Markdown + Code)**  
   - `du.evaluate_precision_recall_at_k(user_emb, movie_emb, test_pairs, k=K)`; print dict.

11) **RMSE (Markdown + Code)**  
   - Fit ridge on train → `du.fit_edge_regressor_ridge(...)`; test RMSE via `du.rmse_from_regressor(...)`.

12) **Top-N recs (Markdown + Code)**  
   - Build seen map, pick a sample user (0), call `du.recommend_topk_for_user(...)`, map to titles via `du.id_maps_to_title_lookup(...)`.

13) **Summary (Code)**  
   - Collect into a small dict: users, movies, edges, P@K, R@K, RMSE, sample user’s top-N titles; print.

14) **Notes & Next Steps (Markdown)**  
   - 5–8 bullets (same as in `DGL.API.md`).
```
