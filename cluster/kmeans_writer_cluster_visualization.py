"""
K-Means writer-cluster membership visualizations: before PCA vs after PCA.

Produces, for both Clustering_v1 and Clustering_v2, four separate images:
  * KMeans_<v>_before_pca_by_writer.png
  * KMeans_<v>_before_pca_by_cluster.png
  * KMeans_<v>_after_pca_by_writer.png
  * KMeans_<v>_after_pca_by_cluster.png
plus:
  * KMeans_<v>_writer_cluster_heatmap.png
      Writer x cluster page-count heatmap, before and after PCA.
  * KMeans_<v>_writer_cluster_membership.csv
      One combined table: every page's writer, cluster and display coords
      for both representations.

Configurations deliberately mirror the saved V1/V2 experiments:
  V1 before PCA : StandardScaler(4096) -> KMeans(K=40)
  V1 after PCA  : StandardScaler(4096) -> PCA(100) -> KMeans(K=40)
  V2 before PCA : raw 4096 features -> KMeans(K=40)
  V2 after PCA  : raw 4096 -> PCA(95% variance) -> KMeans(K=40)

2-D coordinates are for DISPLAY ONLY. For the "before PCA" images the
display PCA(2) is fitted on the same representation used for clustering;
for the "after PCA" images the first two components of the actual PCA are
used.

Output root:
    cluster/kmeans_writer_cluster_plots/

Run:
    python cluster/kmeans_writer_cluster_visualization.py
"""

from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "cluster" / "kmeans_writer_cluster_plots"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

V1_FEATURES = PROJECT_ROOT / "features"
V2_FEATURES = PROJECT_ROOT / "cluster" / "Clustering_v2" / "features"

K = 40
RANDOM_STATE = 42
N_INIT = 20


def load_features(feature_dir, split):
    X = np.load(feature_dir / f"X_{split}_page.npy", allow_pickle=False)
    y = np.asarray(
        np.load(feature_dir / f"y_{split}_page.npy", allow_pickle=True)
    ).reshape(-1)
    return X.astype(np.float64), y


def writer_names_from_metadata(feature_dir, split, y):
    """Map integer writer ids to the letter names when metadata exists."""
    path = feature_dir / f"{split}_page_metadata.csv"
    if path.exists():
        meta = pd.read_csv(path)
        mapping = (
            meta.drop_duplicates("writer_id")
            .set_index("writer_id")["writer"]
            .to_dict()
        )
        return [str(mapping.get(int(w), int(w))) for w in y]
    return [str(int(w)) for w in y]


def build_v1():
    """V1: scaled features before PCA; scaled + PCA(100) after PCA."""
    X, y = load_features(V1_FEATURES, "train")
    names = writer_names_from_metadata(V1_FEATURES, "train", y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    reps = {}

    # Before PCA: cluster on the full scaled feature matrix.
    pca_display = PCA(n_components=2, random_state=RANDOM_STATE)
    xy_before = pca_display.fit_transform(X_scaled)
    reps["before"] = (
        X_scaled,
        xy_before,
        "Before PCA: StandardScaler(4096) + K-Means\n"
        f"(2-D PCA display, {pca_display.explained_variance_ratio_.sum() * 100:.1f}% var)",
    )

    # After PCA: StandardScaler + PCA(100), matching V1 script 06.
    pca = PCA(n_components=100, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    reps["after"] = (
        X_pca,
        X_pca[:, :2],
        "After PCA: StandardScaler + PCA(100) + K-Means\n"
        f"(first 2 of 100 PCs, {pca.explained_variance_ratio_[:2].sum() * 100:.1f}% var)",
    )

    return X, y, names, reps


def build_v2():
    """V2: raw features before PCA; raw + PCA(95%) after PCA (Option B)."""
    X, y = load_features(V2_FEATURES, "train")
    names = writer_names_from_metadata(V2_FEATURES, "train", y)

    reps = {}

    pca_display = PCA(n_components=2, random_state=RANDOM_STATE)
    xy_before = pca_display.fit_transform(X)
    reps["before"] = (
        X,
        xy_before,
        "Before PCA: raw 4096-D features + K-Means\n"
        f"(2-D PCA display, {pca_display.explained_variance_ratio_.sum() * 100:.1f}% var)",
    )

    pca = PCA(n_components=0.95, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X)
    reps["after"] = (
        X_pca,
        X_pca[:, :2],
        f"After PCA: raw features + PCA95 ({X_pca.shape[1]} dims) + K-Means\n"
        f"(first 2 PCs, {pca.explained_variance_ratio_[:2].sum() * 100:.1f}% var)",
    )

    return X, y, names, reps


def cluster_and_plot(version, X_raw, y, names, reps):
    run_name = f"KMeans_{version}"
    out_dir = OUTPUT_ROOT / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    unique_writers = sorted(np.unique(y).tolist())
    n_writers = len(unique_writers)
    writer_color = {}
    tab20 = plt.get_cmap("tab20")
    tab20b = plt.get_cmap("tab20b")
    for rank, w in enumerate(unique_writers):
        cmap = tab20 if rank < 20 else tab20b
        writer_color[w] = cmap(rank % 20)

    results = {}
    rows = []

    for mode, (X_repr, xy, title) in reps.items():
        model = KMeans(
            n_clusters=K,
            random_state=RANDOM_STATE,
            n_init=N_INIT,
        )
        labels = model.fit_predict(X_repr)
        results[mode] = {"labels": labels, "xy": xy, "title": title}

        rows.append(
            pd.DataFrame(
                {
                    "representation": mode,
                    "writer_id": y,
                    "writer": names,
                    "page_index": np.arange(len(y)),
                    "cluster_id": labels,
                    "display_pc1": xy[:, 0],
                    "display_pc2": xy[:, 1],
                }
            )
        )

    # One combined CSV for easy inspection.
    pd.concat(rows, ignore_index=True).to_csv(
        out_dir / f"{run_name}_writer_cluster_membership.csv",
        index=False,
    )

    # --------------------------------------------------------------
    # Separate figures per representation and colouring.
    # --------------------------------------------------------------
    cluster_cmap = plt.get_cmap("turbo")

    for mode in ["before", "after"]:
        labels = results[mode]["labels"]
        xy = results[mode]["xy"]
        mode_label = f"{mode}_pca"

        # 1) Colored by writer, annotate cluster id.
        fig, ax = plt.subplots(figsize=(12, 9))
        colors = [writer_color[w] for w in y]
        ax.scatter(
            xy[:, 0], xy[:, 1], c=colors, s=70, alpha=0.85,
            edgecolors="black", linewidths=0.3,
        )
        for (x, yy), cid in zip(xy, labels):
            ax.annotate(
                str(int(cid)), (x, yy), fontsize=5.5,
                ha="center", va="center",
            )
        ax.set_title(
            f"K-Means {version.upper()}: {results[mode]['title']}\n"
            "Color = writer (person); text = assigned cluster ID",
            fontsize=12,
        )
        ax.set_xlabel("Display PC 1")
        ax.set_ylabel("Display PC 2")
        fig.tight_layout()
        path = out_dir / f"{run_name}_{mode_label}_by_writer.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[{version}] writer-coloured -> {path}")

        # 2) Colored by cluster, annotate writer.
        fig, ax = plt.subplots(figsize=(12, 9))
        scatter = ax.scatter(
            xy[:, 0], xy[:, 1], c=labels, cmap=cluster_cmap,
            vmin=0, vmax=K - 1, s=70, alpha=0.85,
            edgecolors="black", linewidths=0.3,
        )
        for (x, yy), name in zip(xy, names):
            ax.annotate(
                str(name), (x, yy), fontsize=5.5,
                ha="center", va="center",
            )
        ax.set_title(
            f"K-Means {version.upper()}: {results[mode]['title']}\n"
            "Color = cluster ID; text = writer (person)",
            fontsize=12,
        )
        ax.set_xlabel("Display PC 1")
        ax.set_ylabel("Display PC 2")
        fig.colorbar(scatter, ax=ax, label="Cluster ID")
        fig.tight_layout()
        path = out_dir / f"{run_name}_{mode_label}_by_cluster.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[{version}] cluster-coloured -> {path}")

    # --------------------------------------------------------------
    # Writer x cluster heatmaps (page counts)
    # --------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(22, 10))
    for ax, mode in zip(axes, ["before", "after"]):
        labels = results[mode]["labels"]
        table = pd.crosstab(
            pd.Categorical(y, categories=unique_writers),
            pd.Categorical(labels, categories=range(K)),
        )
        im = ax.imshow(table.values, aspect="auto", cmap="viridis")
        ax.set_title(
            f"{'Before' if mode == 'before' else 'After'} PCA: "
            "writer x cluster page counts",
            fontsize=12,
        )
        ax.set_xlabel("Cluster ID")
        ax.set_ylabel("Writer ID")
        ax.set_xticks(range(0, K, 5))
        ax.set_yticks(range(n_writers))
        ax.set_yticklabels(unique_writers, fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        f"K-Means {version.upper()}: writer-cluster membership heatmap, "
        f"before vs after PCA (K={K})",
        fontsize=16,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    heat_path = out_dir / f"{run_name}_writer_cluster_heatmap.png"
    fig.savefig(heat_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[{version}] heatmap -> {heat_path}")


def main():
    print("Building V1 representations...")
    X1, y1, names1, reps1 = build_v1()
    cluster_and_plot("v1", X1, y1, names1, reps1)

    print("Building V2 representations...")
    X2, y2, names2, reps2 = build_v2()
    cluster_and_plot("v2", X2, y2, names2, reps2)

    print(f"\nAll outputs written under: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
