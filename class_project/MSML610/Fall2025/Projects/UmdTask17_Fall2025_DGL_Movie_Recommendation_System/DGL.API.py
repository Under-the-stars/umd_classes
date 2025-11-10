"""
DGL.API.py — Native API + Thin Wrapper Demonstration (MSML610 / MovieLens)

This script demonstrates the *native* DGL pieces we rely on (heterographs,
edge data, GraphSAGE) and the *thin wrapper* functions implemented in
`dgl_utils.py`. It is intentionally small and fast to run on CPU.

What it does:
  1) Loads MovieLens ratings (and movies.csv if available) or a tiny toy sample.
  2) Remaps ids to contiguous ints.
  3) Builds a bipartite heterograph with edge weights (ratings).
  4) Optionally builds movie-genre one-hot features and fuses them.
  5) Splits edges into train/val/test.
  6) Trains a small GraphSAGE link-pred model for a few epochs (didactic).
  7) Evaluates Precision@K / Recall@K and rating RMSE via a ridge regressor.
  8) Produces Top-N recommendations for a sample user.

Usage (CPU):
  python DGL.API.py --ratings data/ratings.csv --movies data/movies.csv ^
                    --max-edges 100000 --epochs 2 --k 10

If --ratings/--movies are omitted or files are missing, a tiny toy dataset
is used so the script can always run to completion.

NOTE: Heavy logic lives in dgl_utils.py; this file is a thin, reproducible
“API tutorial” driver that mirrors the structure used in the class templates.

Author: <your name>
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import dgl  # noqa: F401  # imported to assert presence

import dgl_utils as du


# -----------------------------------------------------------------------------
# Args & logging
# -----------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DGL API tutorial runner")
    p.add_argument("--ratings", type=str, default="", help="Path to ratings.csv")
    p.add_argument("--movies", type=str, default="", help="Path to movies.csv (optional)")
    p.add_argument("--max-edges", type=int, default=100_000, help="Max edges to sample from ratings for speed")
    p.add_argument("--epochs", type=int, default=2, help="Epochs for the didactic link-pred training")
    p.add_argument("--embed-dim", type=int, default=32, help="Embedding dimension")
    p.add_argument("--k", type=int, default=10, help="K for Precision@K / Recall@K and Top-N")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device (API demo works on CPU)")
    p.add_argument("--sample-user", type=int, default=0, help="User index (contiguous id) to show recs for")
    return p.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _init_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger("DGL.API")


# -----------------------------------------------------------------------------
# Data loading (real or toy)
# -----------------------------------------------------------------------------

def _load_or_toy(ratings_path: str, movies_path: str, max_edges: int, seed: int) -> Dict[str, pd.DataFrame]:
    """
    Load ratings (+ movies) if present; otherwise return a tiny toy dataset.
    """
    have_real = ratings_path and os.path.exists(ratings_path)
    if have_real:
        data = du.load_movielens(ratings_path, movies_path if movies_path and os.path.exists(movies_path) else None)
        ratings = data["ratings"]
        if max_edges and len(ratings) > max_edges:
            ratings = ratings.sample(n=max_edges, random_state=seed).reset_index(drop=True)
        out = {"ratings": ratings}
        if "movies" in data:
            out["movies"] = data["movies"]
        return out

    # Toy fallback (always runs).
    ratings = pd.DataFrame({
        "userId":   [10,10,11,12,12,13,13,13],
        "movieId":  [100,101,100,102,103,100,102,104],
        "rating":   [4.0,5.0,3.0,4.5,2.5,4.0,3.5,5.0],
        "timestamp":[1,2,3,4,5,6,7,8],
    })
    movies = pd.DataFrame({
        "movieId": [100,101,102,103,104],
        "title": ["Toy Story", "Jumanji", "Grumpier Old Men", "Waiting to Exhale", "Father of the Bride Part II"],
        "genres": [
            "Adventure|Animation|Children|Comedy|Fantasy",
            "Adventure|Children|Fantasy",
            "Comedy|Romance",
            "Comedy|Drama|Romance",
            "Comedy",
        ],
    })
    return {"ratings": ratings, "movies": movies}


# -----------------------------------------------------------------------------
# Main driver
# -----------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    _set_seed(args.seed)
    log = _init_logger()
    log.info("Starting DGL API demo (device=%s)", args.device)

    # 1) Load data (real or toy)
    data = _load_or_toy(args.ratings, args.movies, args.max_edges, args.seed)
    ratings = data["ratings"]
    movies  = data.get("movies", None)
    log.info("Ratings rows=%d; Movies rows=%s", len(ratings), "n/a" if movies is None else len(movies))

    # 2) Remap ids to contiguous ints
    df2, maps = du.remap_ids(ratings, "userId", "movieId")
    num_users = df2["u"].nunique()
    num_movies = df2["v"].nunique()
    log.info("Contiguous users=%d movies=%d edges=%d", num_users, num_movies, len(df2))

    # 3) Build heterograph with edge features (ratings, timestamp)
    g = du.build_bipartite_graph(df2, num_users, num_movies, rating_col="rating")
    log.info("Graph built: %s | etypes=%s", g.ntypes, g.etypes)

    # 4) Optional: movie genre features (node features)
    movie_feat_tensor = None
    genre_vocab: List[str] = []
    if movies is not None:
        movie_feat_tensor, genre_vocab = du.build_movie_genre_onehot(movies, maps["item_map"])
        log.info("Movie features: shape=%s | #genres=%d", tuple(movie_feat_tensor.shape), len(genre_vocab))

    # 5) Edge splits
    splits = du.make_edge_splits(g, etype=("user", "rates", "movie"), test_size=0.1, val_size=0.1, seed=args.seed)
    train_pairs = du.eids_to_pairs(g, splits["train_eids"])
    val_pairs   = du.eids_to_pairs(g, splits["val_eids"])
    test_pairs  = du.eids_to_pairs(g, splits["test_eids"])

    # Keep ratings per split for RMSE
    r_all = g.edges[('user','rates','movie')].data["rating"].numpy()
    train_ratings = [r_all[i] for i in splits["train_eids"].tolist()]
    val_ratings   = [r_all[i] for i in splits["val_eids"].tolist()]
    test_ratings  = [r_all[i] for i in splits["test_eids"].tolist()]

    log.info("Splits: train=%d val=%d test=%d", len(train_pairs), len(val_pairs), len(test_pairs))

    # 6) Train GraphSAGE link prediction (didactic, few epochs)
    embs = du.train_link_prediction(
        g, splits,
        embed_dim=args.embed_dim,
        epochs=args.epochs,
        lr=1e-3,
        device=args.device,
        movie_feat_tensor=movie_feat_tensor
    )
    user_emb, movie_emb = embs["user_emb"], embs["movie_emb"]
    log.info("Embeddings: users=%s movies=%s", tuple(user_emb.shape), tuple(movie_emb.shape))

    # 7) Evaluate P@K / R@K on test edges
    metrics_k = du.evaluate_precision_recall_at_k(user_emb, movie_emb, test_pairs, k=args.k)
    log.info("P@%d=%.4f | R@%d=%.4f", args.k, metrics_k["precision@k"], args.k, metrics_k["recall@k"])

    # 8) RMSE via ridge regressor on frozen embeddings (train→fit, test→score)
    reg = du.fit_edge_regressor_ridge(user_emb, movie_emb, train_pairs, train_ratings, alpha=1.0)
    rmse_test = du.rmse_from_regressor(reg, user_emb, movie_emb, test_pairs, test_ratings)
    log.info("Rating RMSE (test)=%.4f", rmse_test)

    # 9) Top-N for a sample user (exclude training items; show titles if available)
    seen_map = du.build_user_seen_map(g, splits["train_eids"])
    topn_idx = du.recommend_topk_for_user(
        args.sample_user, user_emb, movie_emb,
        seen_items=seen_map.get(args.sample_user, set()), k=args.k
    )
    title_lookup = du.id_maps_to_title_lookup(movies, maps.get("item_map"))
    recs = [(mid, title_lookup.get(mid, f"movie_{mid}")) for mid in topn_idx]

    # 10) Print compact summary for your write-up
    summary = {
        "users": int(user_emb.shape[0]),
        "movies": int(movie_emb.shape[0]),
        "edges": int(g.num_edges(("user","rates","movie"))),
        f"P@{args.k}": round(metrics_k["precision@k"], 4),
        f"R@{args.k}": round(metrics_k["recall@k"], 4),
        "RMSE": round(rmse_test, 4),
        "sample_user": args.sample_user,
        "topN_titles": [t for (_, t) in recs],
    }
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
