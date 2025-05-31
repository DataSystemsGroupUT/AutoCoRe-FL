import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
def aggregate_kmeans_centroids(client_stats):
    """
    Federated average; **drops empty clusters** (total_count == 0).
    Returns new_centroids and a boolean mask of kept clusters
    so clients can prune consistently.
    """
    sums, counts = zip(*client_stats)
    sums   = np.stack(sums)
    counts = np.stack(counts)

    total_counts = counts.sum(0)
    total_sums   = sums.sum(0)

    with np.errstate(divide="ignore", invalid="ignore"):
        centroids = np.where(total_counts[:, None] > 0,
                             total_sums / np.maximum(total_counts[:, None], 1),
                             0)

    return centroids, total_counts > 0


def pareto_select(df: pd.DataFrame, max_rounds: int = 3) -> pd.DataFrame:
    """
    §8: Iteratively pick Pareto‐optimal rules over (support ↑, precision ↑, complexity ↓):
      - represent each as (sup, prec, -complexity)
      - a rule is dominated if ∃ another≥ on all dims and > on at least one
      - pick non‐dominated, remove them, repeat for up to max_rounds
    """
    selected = []
    remaining = df.copy()

    def is_dominated(v, others):
        # v = (s,p,c), others = array of shape (M,3)
        return np.any(
            (others[:,0] >= v[0]) &
            (others[:,1] >= v[1]) &
            (others[:,2] >= v[2]) &  # since we use -complexity
            (np.any(others != v, axis=1))
        )
    for _ in range(max_rounds):
        if remaining.empty:
            break
        arr = remaining[["support","precision"]].values
        comp = -remaining["complexity"].values[:,None]
        pts = np.hstack([arr, comp])
        mask = []
        for i, v in enumerate(pts):
            mask.append(not is_dominated(v, pts))
        pareto_idxs = remaining.index[mask]
        selected.append(remaining.loc[pareto_idxs])
        remaining = remaining.drop(index=pareto_idxs).reset_index(drop=True)

    if selected:
        return pd.concat(selected, ignore_index=True)
    return df
