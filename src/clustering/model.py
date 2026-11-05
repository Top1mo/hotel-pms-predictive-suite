"""Baseline clustering: KMeans, Agglomerative, GMM with a k sweep.

Run from the project root (hotel-pms-predictive-suite):
    python -m src.clustering.model
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture

from src.data.loader import CATEGORICAL_COLUMNS
from src.preprocessing.pipelines import make_preprocessor, task_data

ROOT = Path(__file__).resolve().parents[2]          # hotel-pms-predictive-suite
TABLES = ROOT / "outputs" / "tables"
SEED = 42

# 1) Load train + validation (test stays reserved)
read = lambda name: pd.read_csv(ROOT / "data" / "splits" / f"{name}.csv", index_col="row_id",
                                dtype={c: "string" for c in CATEGORICAL_COLUMNS})
df = pd.concat([read("train"), read("validation")])

# 2) Preprocess with the clustering pipeline (no target)
X, _ = task_data(df, "clustering")
Z = make_preprocessor("clustering").fit_transform(X)
Z = Z.toarray() if hasattr(Z, "toarray") else np.asarray(Z)

# Agglomerative is slow on all rows, so it uses a random sample of 5,000
Zs = Z[np.random.default_rng(SEED).choice(len(Z), min(5000, len(Z)), replace=False)]

# 3) The three models
models = {
    "KMeans":        (Z,  lambda k: KMeans(n_clusters=k, n_init=10, random_state=SEED)),
    "Agglomerative": (Zs, lambda k: AgglomerativeClustering(n_clusters=k, linkage="ward")),
    "GMM":           (Z,  lambda k: GaussianMixture(n_components=k, random_state=SEED)),
}

# 4) Try k = 2..8 for each model and score it
results = []
for name, (data, build) in models.items():
    for k in range(2, 9):
        labels = build(k).fit_predict(data)
        results.append({
            "model": name, "k": k,
            "silhouette": silhouette_score(data, labels, sample_size=10000, random_state=SEED),
            "davies_bouldin": davies_bouldin_score(data, labels),
        })
    print(name, "done")

results = pd.DataFrame(results).round(3)
print(results.to_string(index=False))

# 5) Baseline (k=3) vs tuned (best silhouette) for each model
baseline = results[results["k"] == 3].assign(config="baseline")
tuned = results.loc[results.groupby("model")["silhouette"].idxmax()].assign(config="tuned")
print("\n", pd.concat([baseline, tuned]).sort_values("model").to_string(index=False))

TABLES.mkdir(parents=True, exist_ok=True)
results.to_csv(TABLES / "clustering_results.csv", index=False)