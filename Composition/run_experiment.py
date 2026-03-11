import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import umap
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------
# Project path setup
# ------------------------------------------------------------
sys.path.append("..")
sys.path.append(os.path.abspath("."))

# ------------------------------------------------------------
# Project imports
# ------------------------------------------------------------
from Composition.encoders import QwenCompositionEncoder, QwenElementEncoder
from Composition.common_utils import *
from Composition.save_utils import load_embeddings
from Composition.chem_parse import *


# ============================================================
# Encoding utilities
# ============================================================
def encode_in_batches(encoder, data, batch_size=64):
    all_emb = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        all_emb.append(encoder.encode(batch))
    return np.vstack(all_emb)


# ============================================================
# Space correlation
# ============================================================


def compute_space_correlation(E1, E2, n_pairs=200000):
    N = E1.shape[0]

    idx_i = np.random.randint(0, N, n_pairs)
    idx_j = np.random.randint(0, N, n_pairs)

    sim1 = np.sum(E1[idx_i] * E1[idx_j], axis=1)
    sim2 = np.sum(E2[idx_i] * E2[idx_j], axis=1)

    corr, _ = pearsonr(sim1, sim2)
    return float(corr)


# ============================================================
# Retrieval evaluation
# ============================================================
def evaluate_retrieval(E_train_elem, E_test_elem,
                       E_train_comp, E_test_comp,
                       X_train, X_test,
                       top_k=5, n_queries=300):

    idxs = np.random.choice(len(X_test), size=min(n_queries, len(X_test)), replace=False)

    elem_acc, comp_acc, overlap = [], [], []

    for i in idxs:
        query = X_test[i]

        q_elem = E_test_elem[i].reshape(1, -1)
        q_comp = E_test_comp[i].reshape(1, -1)

        cos_elem = cosine_search(q_elem, E_train_elem, X_train, top_k)
        cos_comp = cosine_search(q_comp, E_train_comp, X_train, top_k)

        e_acc, _ = compute_accuracy(query, list(cos_elem["composition"]))
        c_acc, _ = compute_accuracy(query, list(cos_comp["composition"]))

        elem_acc.append(e_acc)
        comp_acc.append(c_acc)

        overlap.append(
            len(set(cos_elem["composition"]) & set(cos_comp["composition"])) / top_k
        )

    return {
        "elem_acc": np.array(elem_acc),
        "comp_acc": np.array(comp_acc),
        "overlap": np.array(overlap),
    }


# ============================================================
# Clustering comparison
# ============================================================
def evaluate_clustering(E_elem, E_comp, n_clusters=20, n_samples=5000):

    if len(E_elem) > n_samples:
        idxs = np.random.choice(len(E_elem), n_samples, replace=False)
        E_elem = E_elem[idxs]
        E_comp = E_comp[idxs]

    labels_elem = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto").fit_predict(E_elem)
    labels_comp = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto").fit_predict(E_comp)

    ami = adjusted_mutual_info_score(labels_elem, labels_comp)
    return float(ami)


# ============================================================
# Plotting utilities
# ============================================================
def plot_accuracy_bar(elem_acc, comp_acc, top_k, out_path):
    means = [elem_acc.mean(), comp_acc.mean()]
    stds = [elem_acc.std(), comp_acc.std()]
    labels = ["Element encoder", "Composition encoder"]

    plt.figure(figsize=(5, 4))
    plt.bar(labels, means, yerr=stds, capsize=6)
    plt.ylabel(f"Chemical accuracy@{top_k}")
    plt.ylim(0, 1)
    plt.title("Retrieval accuracy comparison")

    for i, v in enumerate(means):
        plt.text(i, v + 0.02, f"{v:.2f}", ha="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_tsne_magpie_vs_qwen_pca(
    magpie_path,
    qwen_elem_path,
    out_path,
    n_samples=10000,
    pca_dim=50,
    perplexity=30,
    random_state=42,
):

    E_mag, _, X_mag, _ = load_embeddings(magpie_path)
    E_qwen, _, X_qwen, _ = load_embeddings(qwen_elem_path)

    mag_dict = dict(zip(map(str, X_mag), E_mag))
    qwen_dict = dict(zip(map(str, X_qwen), E_qwen))

    common = sorted(set(mag_dict) & set(qwen_dict))

    if len(common) == 0:
        raise ValueError("No common compositions found.")

    E_mag = normalize(np.vstack([mag_dict[k] for k in common]))
    E_qwen = normalize(np.vstack([qwen_dict[k] for k in common]))

    rng = np.random.default_rng(random_state)

    if len(E_mag) > n_samples:
        idx = rng.choice(len(E_mag), n_samples, replace=False)
        E_mag = E_mag[idx]
        E_qwen = E_qwen[idx]

    # PCA separately
    pca_mag = PCA(n_components=pca_dim, random_state=random_state)
    pca_qwen = PCA(n_components=pca_dim, random_state=random_state)

    Z_mag = pca_mag.fit_transform(E_mag)
    Z_qwen = pca_qwen.fit_transform(E_qwen)

    X_joint = np.vstack([Z_mag, Z_qwen])

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    )

    coords = tsne.fit_transform(X_joint)

    n = len(Z_mag)

    plt.figure(figsize=(10,10))

    # MAGPIE points
    plt.scatter(
        coords[:n,0], coords[:n,1],
        s=25,
        alpha=0.7,
        color="royalblue",
        label="Magpie → Qwen space"
    )

    # QWEN points
    plt.scatter(
        coords[n:,0], coords[n:,1],
        s=25,
        alpha=0.7,
        color="darkorange",
        label="Qwen (original)"
    )

    plt.legend(fontsize=22, frameon=False)

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    plt.tight_layout()

    plt.savefig(out_path, dpi=400, format="pdf")
    plt.close()






def plot_umap_and_workspace_correlation(
    magpie_path,
    qwen_path,
    max_samples=2000,
    umap_neighbors=15,
    umap_min_dist=0.1,
    save_path=None,
    random_state=42,
):
    """
    Correct comparison:
    - UMAP(Magpie) and UMAP(Qwen) separately
    - Workspace (similarity-space) correlation quantitatively
    """

    # --------------------------------------------------
    # 1. Load embeddings
    # --------------------------------------------------
    E_mag, _, X_mag, _ = load_embeddings(magpie_path)
    E_qwen, _, X_qwen, _ = load_embeddings(qwen_path)

    # --------------------------------------------------
    # 2. Align by composition
    # --------------------------------------------------
    mag_dict = dict(zip(map(str, X_mag), E_mag))
    qwen_dict = dict(zip(map(str, X_qwen), E_qwen))

    common = sorted(set(mag_dict) & set(qwen_dict))
    if len(common) == 0:
        raise ValueError("No common compositions found.")

    E_mag = normalize(np.vstack([mag_dict[k] for k in common]))
    E_qwen = normalize(np.vstack([qwen_dict[k] for k in common]))

    # --------------------------------------------------
    # 3. Subsample (shared indices!)
    # --------------------------------------------------
    rng = np.random.default_rng(random_state)
    if len(E_mag) > max_samples:
        idx = rng.choice(len(E_mag), max_samples, replace=False)
        E_mag = E_mag[idx]
        E_qwen = E_qwen[idx]

    # --------------------------------------------------
    # 4. UMAP (SEPARATELY)
    # --------------------------------------------------
    reducer_mag = umap.UMAP(
        n_neighbors=umap_neighbors,
        min_dist=umap_min_dist,
        random_state=random_state,
    )
    reducer_qwen = umap.UMAP(
        n_neighbors=umap_neighbors,
        min_dist=umap_min_dist,
        random_state=random_state,
    )

    umap_mag = reducer_mag.fit_transform(E_mag)
    umap_qwen = reducer_qwen.fit_transform(E_qwen)

    # --------------------------------------------------
    # 5. Workspace correlation
    # --------------------------------------------------
    S_mag = cosine_similarity(E_mag)
    S_qwen = cosine_similarity(E_qwen)

    mask = ~np.eye(S_mag.shape[0], dtype=bool)
    corr, _ = pearsonr(S_mag[mask], S_qwen[mask])

    # --------------------------------------------------
    # 6. Plot (3 panels)
    # --------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # --- UMAP Magpie ---
    axes[0].scatter(umap_mag[:, 0], umap_mag[:, 1], s=6, alpha=0.6)
    axes[0].set_title("UMAP – Magpie")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # --- UMAP Qwen ---
    axes[1].scatter(umap_qwen[:, 0], umap_qwen[:, 1], s=6, alpha=0.6)
    axes[1].set_title("UMAP – Qwen")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    # --- Workspace correlation ---
    axes[2].scatter(S_mag[mask], S_qwen[mask], s=2, alpha=0.3)
    axes[2].set_xlabel("Magpie cosine similarity")
    axes[2].set_ylabel("Qwen cosine similarity")
    axes[2].set_title(f"Workspace correlation (ρ = {corr:.3f})")
    axes[2].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        #print(f"[Saved] UMAP + workspace correlation → {save_path}")

    plt.show()

    return float(corr)



def plot_tsne(
    E_elem,
    E_comp,
    out_path,
    n_samples=8000,
    perplexity=40,
    random_state=42,
):

    rng = np.random.default_rng(random_state)

    # Subsample for clarity
    if len(E_elem) > n_samples:
        idx = rng.choice(len(E_elem), n_samples, replace=False)
        E_elem = E_elem[idx]
        E_comp = E_comp[idx]

    # Stack embeddings
    X = np.vstack([E_elem, E_comp])

    # Standardize
    X = StandardScaler().fit_transform(X)

    # PCA pre-reduction
    if X.shape[1] > 50:
        X = PCA(n_components=50, random_state=random_state).fit_transform(X)

    # t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        n_iter=1500,
        random_state=random_state,
    )

    coords = tsne.fit_transform(X)
    n = len(E_elem)

    plt.figure(figsize=(10, 10), dpi=400)

    # Element-level embeddings
    plt.scatter(
        coords[:n, 0],
        coords[:n, 1],
        s=25,
        alpha=0.7,
        color="royalblue",
        label="Element-level",
    )

    # Composition-level embeddings
    plt.scatter(
        coords[n:, 0],
        coords[n:, 1],
        s=25,
        alpha=0.7,
        color="darkorange",
        label="Composition-level",
    )

    plt.xticks([])
    plt.yticks([])

    plt.legend(fontsize=20, frameon=False)

    plt.tight_layout()

    plt.savefig(out_path, format="pdf", bbox_inches="tight")

    plt.close()
# ============================================================
# Main experiment
# ============================================================
def run_experiment(elem_path, comp_path, out_dir="results", top_k=5):

    os.makedirs(out_dir, exist_ok=True)

    # Load & align
    E_train_elem, E_test_elem, X_train_elem, X_test_elem = load_embeddings(elem_path)
    E_train_comp, E_test_comp, X_train_comp, X_test_comp = load_embeddings(comp_path)

    # Align by composition string
    train_dict_elem = dict(zip(map(str, X_train_elem), E_train_elem))
    train_dict_comp = dict(zip(map(str, X_train_comp), E_train_comp))

    common = sorted(set(train_dict_elem) & set(train_dict_comp))
    E_train_elem = normalize(np.vstack([train_dict_elem[k] for k in common]))
    E_train_comp = normalize(np.vstack([train_dict_comp[k] for k in common]))
    X_train = common

    # Test sets
    test_dict_elem = dict(zip(map(str, X_test_elem), E_test_elem))
    test_dict_comp = dict(zip(map(str, X_test_comp), E_test_comp))

    common_test = sorted(set(test_dict_elem) & set(test_dict_comp))
    E_test_elem = normalize(np.vstack([test_dict_elem[k] for k in common_test]))
    E_test_comp = normalize(np.vstack([test_dict_comp[k] for k in common_test]))
    X_test = common_test

    # Metrics
    space_corr = compute_space_correlation(E_train_elem, E_train_comp)
    retrieval = evaluate_retrieval(
        E_train_elem, E_test_elem,
        E_train_comp, E_test_comp,
        X_train, X_test,
        top_k=top_k
    )
    ami = evaluate_clustering(E_train_elem, E_train_comp)

    # Plots
    plot_accuracy_bar(
        retrieval["elem_acc"], retrieval["comp_acc"],
        top_k, os.path.join(out_dir, "accuracy_bar.png")
    )

    #plot_tsne(
     #   E_train_elem, E_train_comp,
      #  os.path.join(out_dir, "tsne.png")
    #)
    plot_tsne(
        E_train_elem,
        E_train_comp,
        os.path.join(out_dir, "tsne_paper.pdf")
    )

    # Save results
    results = {
        "top_k": top_k,
        "space_correlation": space_corr,
        "element_accuracy_mean": float(retrieval["elem_acc"].mean()),
        "composition_accuracy_mean": float(retrieval["comp_acc"].mean()),
        "element_accuracy_std": float(retrieval["elem_acc"].std()),
        "composition_accuracy_std": float(retrieval["comp_acc"].std()),
        "mean_overlap": float(retrieval["overlap"].mean()),
        "clustering_ami": ami,
    }

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


def compare_magpie_vs_qwen(
    magpie_path,
    qwen_elem_path,
    top_k=5,
    n_queries=300
):
    # Load saved embeddings
    E_train_mag, E_test_mag, X_train_mag, X_test_mag = load_embeddings(magpie_path)
    E_train_qwen, E_test_qwen, X_train_qwen, X_test_qwen = load_embeddings(qwen_elem_path)

    # Align by composition string
    train_mag = dict(zip(map(str, X_train_mag), E_train_mag))
    train_qwen = dict(zip(map(str, X_train_qwen), E_train_qwen))

    common_train = sorted(set(train_mag) & set(train_qwen))
    X_train = common_train

    E_train_mag = normalize(np.vstack([train_mag[k] for k in common_train]))
    E_train_qwen = normalize(np.vstack([train_qwen[k] for k in common_train]))

    # Same for test
    test_mag = dict(zip(map(str, X_test_mag), E_test_mag))
    test_qwen = dict(zip(map(str, X_test_qwen), E_test_qwen))

    common_test = sorted(set(test_mag) & set(test_qwen))
    X_test = common_test

    E_test_mag = normalize(np.vstack([test_mag[k] for k in common_test]))
    E_test_qwen = normalize(np.vstack([test_qwen[k] for k in common_test]))

    # Evaluate retrieval
    idxs = np.random.choice(len(X_test), size=min(n_queries, len(X_test)), replace=False)

    mag_acc, qwen_acc = [], []

    for i in idxs:
        query = X_test[i]

        q_mag = E_test_mag[i].reshape(1, -1)
        q_qwen = E_test_qwen[i].reshape(1, -1)

        res_mag = cosine_search(q_mag, E_train_mag, X_train, top_k)
        res_qwen = cosine_search(q_qwen, E_train_qwen, X_train, top_k)

        acc_mag, _ = compute_accuracy(query, res_mag["composition"])
        acc_qwen, _ = compute_accuracy(query, res_qwen["composition"])

        mag_acc.append(acc_mag)
        qwen_acc.append(acc_qwen)

    return {
        "magpie_accuracy_mean": float(np.mean(mag_acc)),
        "magpie_accuracy_std": float(np.std(mag_acc)),
        "qwen_accuracy_mean": float(np.mean(qwen_acc)),
        "qwen_accuracy_std": float(np.std(qwen_acc)),
    }



def plot_workspace_correlation(
    magpie_path,
    qwen_path,
    max_samples=1500,
    save_path=None,
    random_state=42,
):
    """
    Load Magpie and Qwen embeddings, align by composition,
    compute workspace (vector-space) correlation, and plot it.

    Parameters
    ----------
    magpie_path : str
        Path to saved Magpie embeddings
    qwen_path : str
        Path to saved Qwen embeddings
    max_samples : int
        Subsample size for plotting (keeps plot readable)
    save_path : str or None
        If provided, saves the figure
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    corr : float
        Pearson correlation between similarity matrices
    """

    # --------------------------------------------------
    # 1. Load embeddings
    # --------------------------------------------------
    E_train_mag, _, X_train_mag, _ = load_embeddings(magpie_path)
    E_train_qwen, _, X_train_qwen, _ = load_embeddings(qwen_path)

    # --------------------------------------------------
    # 2. Align by composition string
    # --------------------------------------------------
    mag_dict = dict(zip(map(str, X_train_mag), E_train_mag))
    qwen_dict = dict(zip(map(str, X_train_qwen), E_train_qwen))

    common = sorted(set(mag_dict) & set(qwen_dict))
    if len(common) == 0:
        raise ValueError("No common compositions found between embeddings.")

    E_mag = normalize(np.vstack([mag_dict[k] for k in common]))
    E_qwen = normalize(np.vstack([qwen_dict[k] for k in common]))

    # --------------------------------------------------
    # 3. Subsample (optional, for large datasets)
    # --------------------------------------------------
    rng = np.random.default_rng(random_state)
    if len(E_mag) > max_samples:
        idx = rng.choice(len(E_mag), max_samples, replace=False)
        E_mag = E_mag[idx]
        E_qwen = E_qwen[idx]

    # --------------------------------------------------
    # 4. Compute similarity matrices
    # --------------------------------------------------
    S_mag = cosine_similarity(E_mag)
    S_qwen = cosine_similarity(E_qwen)

    mask = ~np.eye(S_mag.shape[0], dtype=bool)
    sim_mag = S_mag[mask]
    sim_qwen = S_qwen[mask]

    # --------------------------------------------------
    # 5. Correlation
    # --------------------------------------------------
    corr, _ = pearsonr(sim_mag, sim_qwen)

    # --------------------------------------------------
    # 6. Plot correlation workspace
    # --------------------------------------------------
    plt.figure(figsize=(5, 5))
    plt.scatter(sim_mag, sim_qwen, s=2, alpha=0.3)
    plt.xlabel("Magpie cosine similarity")
    plt.ylabel("Qwen cosine similarity")
    plt.title(f"Workspace correlation (ρ = {corr:.3f})")
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        #print(f"[Saved] Workspace correlation plot → {save_path}")

    plt.show()

    return float(corr)


def save_results_table(results_dict, out_dir, filename):
    """
    Save results as CSV and LaTeX table in a given directory.
    """
    os.makedirs(out_dir, exist_ok=True)

    df = pd.DataFrame([results_dict])

    csv_path = os.path.join(out_dir, f"{filename}.csv")
    tex_path = os.path.join(out_dir, f"{filename}.tex")

    df.to_csv(csv_path, index=False)

    with open(tex_path, "w") as f:
        f.write(
            df.to_latex(
                index=False,
                float_format="%.3f",
                caption="Retrieval performance comparison.",
                label=f"tab:{filename}"
            )
        )

    #print(f"[Saved] CSV table → {csv_path}")
    #print(f"[Saved] LaTeX table → {tex_path}")