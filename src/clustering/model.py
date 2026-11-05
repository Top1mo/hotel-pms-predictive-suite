"""Clustering: KMeans, Agglomerative, GMM, each baseline vs tuned.

Run from the project root (hotel-pms-predictive-suite):
    python -m src.clustering.model

Outputs:
    outputs/tables/clustering_all_runs.csv         every setting tried
    outputs/tables/clustering_baseline_vs_tuned.csv the table for the report
    outputs/tables/clustering_profiles_k*.csv      cluster descriptions
    outputs/figures/clustering_silhouette_vs_k.png
    outputs/figures/clustering_dendrogram.png
    outputs/figures/clustering_pca_k*.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # save plots to files, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture

from src.data.loader import CATEGORICAL_COLUMNS
from src.preprocessing.pipelines import make_preprocessor, task_data

ROOT = Path(__file__).resolve().parents[2]          # hotel-pms-predictive-suite
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"

SEED = 42
K_VALUES = range(2, 9)
AGGLO_SAMPLE = 5000         # hierarchical clustering is too slow on all rows
MIN_CLUSTER_FRAC = 0.02     # "tuned" must not just split off a tiny group
INTERPRET_K = 6             # extra KMeans k to describe, besides the tuned one

# ---------------------------------------------------------------- 1) Data
read = lambda name: pd.read_csv(ROOT / "data" / "splits" / f"{name}.csv", index_col="row_id",
                                dtype={c: "string" for c in CATEGORICAL_COLUMNS})
df = pd.concat([read("train"), read("validation")])      # test stays reserved

X, _ = task_data(df, "clustering")                       # no target
Z = make_preprocessor("clustering").fit_transform(X)
Z = Z.toarray() if hasattr(Z, "toarray") else np.asarray(Z)
Zs = Z[np.random.default_rng(SEED).choice(len(Z), min(AGGLO_SAMPLE, len(Z)), replace=False)]
print(f"rows={len(df)}  features={Z.shape[1]}")

# ---------------------------------------------------------------- 2) Models + search spaces
# name: (data, baseline params, parameter to tune besides k, its values, builder)
models = {
    "KMeans": (Z, {"k": 3, "param": "k-means++"}, "init", ["k-means++", "random"],
               lambda k, p: KMeans(n_clusters=k, init=p, n_init=10, random_state=SEED)),
    "Agglomerative": (Zs, {"k": 3, "param": "ward"}, "linkage", ["ward", "average", "complete"],
                      lambda k, p: AgglomerativeClustering(n_clusters=k, linkage=p)),
    "GMM": (Z, {"k": 3, "param": "full"}, "covariance_type", ["full", "diag", "tied"],
            lambda k, p: GaussianMixture(n_components=k, covariance_type=p, random_state=SEED)),
}

# ---------------------------------------------------------------- 3) Run every setting
runs = []
for name, (data, base, param_name, values, build) in models.items():
    for p in values:
        for k in K_VALUES:
            labels = build(k, p).fit_predict(data)
            sizes = np.bincount(labels, minlength=k)
            runs.append({
                "model": name, "k": k, "param_name": param_name, "param": p,
                "silhouette": silhouette_score(data, labels, sample_size=10000, random_state=SEED),
                "davies_bouldin": davies_bouldin_score(data, labels),
                "smallest_cluster": sizes.min() / len(labels),
            })
    print(name, "done")

runs = pd.DataFrame(runs).round(4)
runs.to_csv(TABLES / "clustering_all_runs.csv", index=False)

# ---------------------------------------------------------------- 4) Baseline vs tuned
summary = []
for name, (_, base, _, _, _) in models.items():
    r = runs[runs["model"] == name]
    b = r[(r["k"] == base["k"]) & (r["param"] == base["param"])].iloc[0]
    valid = r[r["smallest_cluster"] >= MIN_CLUSTER_FRAC]          # every cluster >= 2% of rows
    t = valid.sort_values("silhouette", ascending=False).iloc[0]
    summary += [b.to_dict() | {"config": "baseline"}, t.to_dict() | {"config": "tuned"}]

summary = pd.DataFrame(summary)[["model", "config", "k", "param_name", "param",
                                 "silhouette", "davies_bouldin", "smallest_cluster"]]
summary.to_csv(TABLES / "clustering_baseline_vs_tuned.csv", index=False)
print("\n" + summary.to_string(index=False))

# ---------------------------------------------------------------- 5) Silhouette vs k plot
fig, ax = plt.subplots(figsize=(8, 4.5))
for (name, p), g in runs.groupby(["model", "param"]):
    ax.plot(g["k"], g["silhouette"], marker="o", label=f"{name} ({p})")
ax.set_xlabel("number of clusters k"); ax.set_ylabel("silhouette (higher is better)")
ax.set_title("Silhouette vs k"); ax.legend(fontsize=7, ncol=2)
fig.tight_layout(); fig.savefig(FIGURES / "clustering_silhouette_vs_k.png", dpi=150); plt.close(fig)

# ---------------------------------------------------------------- 6) Dendrogram
fig = plt.figure(figsize=(10, 4))
dendrogram(linkage(Zs[:1000], method="ward"), truncate_mode="lastp", p=30)
plt.title("Hierarchical clustering dendrogram (1,000-row random sample, ward)")
plt.xlabel("merged groups (size in brackets)"); plt.ylabel("distance")
fig.tight_layout(); fig.savefig(FIGURES / "clustering_dendrogram.png", dpi=150); plt.close(fig)

# ---------------------------------------------------------------- 7) PCA plots + profiles
tuned = summary[(summary["model"] == "KMeans") & (summary["config"] == "tuned")].iloc[0]
pca = PCA(n_components=2, random_state=SEED).fit(Z)
P = pca.transform(Z)
print(f"\nPCA 2D explains {pca.explained_variance_ratio_.sum():.1%} of the variance")

for k, init in {int(tuned["k"]): tuned["param"], INTERPRET_K: "k-means++"}.items():
    labels = KMeans(n_clusters=k, init=init, n_init=10, random_state=SEED).fit_predict(Z)

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(P[:, 0], P[:, 1], c=labels, s=2, cmap="tab10", alpha=0.5)
    ax.legend(*sc.legend_elements(), title="cluster", fontsize=7)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title(f"KMeans k={k} in PCA space")
    fig.tight_layout(); fig.savefig(FIGURES / f"clustering_pca_k{k}.png", dpi=150); plt.close(fig)

    # Outcomes (ADR, IsCanceled) are NOT used for clustering, only to describe clusters afterwards
    profile = df.assign(cluster=labels).groupby("cluster").agg(
        size=("LeadTime", "size"),
        lead_time=("LeadTime", "mean"),
        weekend_nights=("StaysInWeekendNights", "mean"),
        week_nights=("StaysInWeekNights", "mean"),
        adults=("Adults", "mean"),
        children=("Children", "mean"),
        repeated_guest=("IsRepeatedGuest", "mean"),
        has_company=("Company", lambda s: s.notna().mean()),
        top_segment=("MarketSegment", lambda s: s.mode().iat[0]),
        top_customer=("CustomerType", lambda s: s.mode().iat[0]),
        adr_after=("ADR", "mean"),
        cancel_rate_after=("IsCanceled", "mean"),
    ).round(2)
    profile.to_csv(TABLES / f"clustering_profiles_k{k}.csv")
    print(f"\nKMeans k={k} cluster profiles:\n" + profile.to_string())

print(f"\nSaved tables to {TABLES}\nSaved figures to {FIGURES}")