import os
import sys
sys.path.append("..")
from Composition.common_utils import *
from Composition.encoders import QwenCompositionEncoder, QwenElementEncoder
from Composition.save_utils import save_embeddings
import numpy as np
from Composition.save_utils import load_embeddings
from Composition.common_utils import cosine_search, compute_accuracy
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
sys.path.append(os.path.abspath("."))
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from Composition.encoders import *
import json
from Composition.common_utils import (load_composition_dataset,cosine_search,compute_accuracy)

# ==============================================================
# STRING-LEVEL QWEN PIPELINE
# ==============================================================

def encode_in_batches(encoder, data, batch_size=64):
    """
    Efficiently encode text data in batches using the provided encoder.
    """
    all_emb = []

    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        batch_emb = encoder.encode(batch)
        all_emb.append(batch_emb)

    return np.vstack(all_emb)

def run_qwen_composition_pipeline(X_train, X_test, top_k=5, save_path=None):

    encoder = QwenCompositionEncoder()

    X_train_emb = encode_in_batches(encoder, X_train, batch_size=64)
    X_test_emb = encode_in_batches(encoder, X_test, batch_size=64)

    X_train_norm = normalize(X_train_emb)
    X_test_norm = normalize(X_test_emb)

    query = X_test[0]
    query_clean = clean_formula_chemically_no_space(query)

    query_emb = encoder.encode([query_clean])
    query_norm = normalize(query_emb)
    query_vec = query_norm.reshape(1, -1)

    cosine_df = cosine_search(query_vec, X_train_norm, X_train, top_k)

    if save_path:
        save_embeddings(save_path, X_train_emb, X_test_emb, X_train, X_test)

    return {
        "query": query,
        "cosine": cosine_df
    }



# ==============================================================
# ELEMENT-LEVEL QWEN PIPELINE
# ==============================================================
def run_qwen_element_pipeline(X_train, X_test, top_k=5, save_path=None):

    encoder = QwenElementEncoder()

    # Encode
    X_train_emb = encoder.encode(X_train)
    X_test_emb = encoder.encode(X_test)

    encoder.save_cache()

    # Normalize
    X_train_norm = normalize(X_train_emb)
    X_test_norm = normalize(X_test_emb)

    # Query
    query = X_test[0]
    query_vec = X_test_norm[0].reshape(1, -1)

    # Search
    cosine_df = cosine_search(query_vec, X_train_norm, X_train, top_k)

    # 🔹 Remove self-match
    cosine_df = cosine_df[cosine_df["composition"] != query].reset_index(drop=True)

    # Accuracy
    cosine_acc, cosine_scores = compute_accuracy(
        query, cosine_df["composition"]
    )

    # 🔹 Clean formatting
    cosine_df["rank"] = np.arange(1, len(cosine_df) + 1)
    cosine_df["score"] = cosine_df["score"].round(4)
    cosine_df = cosine_df[["rank", "composition", "score"]]

    # Save
    if save_path:
        save_embeddings(save_path, X_train_emb, X_test_emb, X_train, X_test)

    return {
        "query": normalize_formula(query),
        "cosine": cosine_df,
        "cosine_accuracy": cosine_acc,
        "cosine_scores": cosine_scores,
    }



def encode_all_compositions_once(compositions, save_path="saved/all_composition_embeddings.npy"):
    """
    Encode ALL compositions in exact given order using QwenElementEncoder.
    The saved embeddings can always be mapped back using keep_idx, train_idx, val_idx, test_idx.
    """

    print(f"[INFO] Encoding {len(compositions)} compositions...")

    encoder = QwenElementEncoder()

    # Encode in exact order — no splits, no shuffling
    embeddings = encoder.encode(compositions)
    encoder.save_cache()

    embeddings = embeddings.astype("float32")

    # Save the aligned embedding matrix
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, embeddings)

    #print(f"[OK] Saved aligned embeddings → {save_path}")
    #print(f"[SHAPE] {embeddings.shape}")

    return embeddings

# ==============================================================
# MAGPIE PIPELINE
# ==============================================================
def run_magpie_pipeline(X_train, X_test, top_k=5, save_path=None):

    encoder = MagpieCompositionEncoder()

    # Encode
    X_train_emb = encoder.fit_transform(X_train)
    X_test_emb = encoder.transform(X_test)

    # Normalize
    X_train_norm = normalize(X_train_emb)
    X_test_norm = normalize(X_test_emb)

    # Query
    query = X_test[0]
    query_vec = X_test_norm[0].reshape(1, -1)

    # Search
    cosine_df = cosine_search(query_vec, X_train_norm, X_train, top_k)

    # 🔹 Remove self-match
    cosine_df = cosine_df[cosine_df["composition"] != query].reset_index(drop=True)

    cosine_acc, cosine_scores = compute_accuracy(
        query, cosine_df["composition"]
    )

    # 🔹 Clean formatting
    cosine_df["rank"] = np.arange(1, len(cosine_df) + 1)
    cosine_df["score"] = cosine_df["score"].round(4)
    cosine_df = cosine_df[["rank", "composition", "score"]]

    # Save embeddings
    if save_path:
        save_embeddings(save_path, X_train_emb, X_test_emb, X_train, X_test)

    return {
        "query": normalize_formula(query),
        "cosine": cosine_df,
        "cosine_accuracy": cosine_acc,
        "cosine_scores": cosine_scores,
    }


def save_all_composition_embeddings(
        compositions,
        save_dir="saved",
        fname="composition_embeddings.npy"
    ):
    """
    Encode ALL compositions in their exact original order using QwenElementEncoder.
    Saves:
        1) embeddings.npy  (float32 matrix, N × D)
        2) metadata.json   (stores composition strings and their positions)
    This ensures perfect mapping with keep_idx and train/val/test splits.
    """

    os.makedirs(save_dir, exist_ok=True)
    emb_path = os.path.join(save_dir, fname)
    meta_path = os.path.join(save_dir, "composition_metadata.json")

    print(f"[INFO] Encoding {len(compositions)} compositions...")

    encoder = QwenElementEncoder()

    #  Encode everything in EXACT order (no shuffling!)
    embeddings = encoder.encode(compositions).astype("float32")
    encoder.save_cache()

    # Save embedding matrix
    np.save(emb_path, embeddings)
    #print(f"[OK] Saved embeddings → {emb_path}  shape={embeddings.shape}")

    # Also save metadata for safety
    metadata = {
        "num_compositions": len(compositions),
        "compositions": list(compositions),
        "embedding_file": fname,
        "dimension": int(embeddings.shape[1])
    }

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    #print(f"[OK] Saved composition metadata → {meta_path}")

    return embeddings


def compute_space_correlation(E_comp, E_elem):
    """
    Computes Pearson correlation between the similarity
    matrices of two embedding spaces.

    Parameters
    ----------
    E_comp : np.ndarray
        (N, d) embeddings from composition-level Qwen encoder
    E_elem : np.ndarray
        (N, d) embeddings from element-level Qwen encoder

    Returns
    -------
    corr : float
        Pearson correlation between the flattened similarity matrices
    """

    # 1) Compute cosine similarity matrices
    S_comp = cosine_similarity(E_comp)
    S_elem = cosine_similarity(E_elem)

    # 2) Flatten (ignore diagonal by masking)
    mask = ~np.eye(S_comp.shape[0], dtype=bool)

    v1 = S_comp[mask]
    v2 = S_elem[mask]

    # 3) Pearson correlation
    corr, _ = pearsonr(v1, v2)
    return float(corr)


def compare_encoders(elem_path, comp_path, top_k=5):

    # 1. Load data
    E_train_elem, E_test_elem, X_train_elem, X_test_elem = load_embeddings(elem_path)
    E_train_comp, E_test_comp, X_train_comp, X_test_comp = load_embeddings(comp_path)

    # 2. Auto-align by composition string
    train_elem_dict = {str(x): emb for x, emb in zip(X_train_elem, E_train_elem)}
    train_comp_dict = {str(x): emb for x, emb in zip(X_train_comp, E_train_comp)}

    test_elem_dict = {str(x): emb for x, emb in zip(X_test_elem, E_test_elem)}
    test_comp_dict = {str(x): emb for x, emb in zip(X_test_comp, E_test_comp)}

    common_train = sorted(set(train_elem_dict.keys()) & set(train_comp_dict.keys()))
    common_test = sorted(set(test_elem_dict.keys()) & set(test_comp_dict.keys()))

    print(f"Common train samples: {len(common_train)}")
    print(f"Common test samples: {len(common_test)}")

    # rebuild aligned arrays
    E_train_elem = np.vstack([train_elem_dict[k] for k in common_train])
    E_train_comp = np.vstack([train_comp_dict[k] for k in common_train])
    X_train = common_train

    E_test_elem = np.vstack([test_elem_dict[k] for k in common_test])
    E_test_comp = np.vstack([test_comp_dict[k] for k in common_test])
    X_test = common_test

    # 3. Normalize
    E_train_elem_norm = normalize(E_train_elem)
    E_train_comp_norm = normalize(E_train_comp)

    # 4. Space correlation
    space_corr = compute_space_correlation(E_train_comp_norm, E_train_elem_norm)

    # 5. Choose same query
    query = X_test[0]
    q_elem = E_test_elem[0].reshape(1, -1)
    q_comp = E_test_comp[0].reshape(1, -1)

    # 6. Search
    cos_elem = cosine_search(q_elem, E_train_elem_norm, X_train, top_k)


    cos_comp = cosine_search(q_comp, E_train_comp_norm, X_train, top_k)

    # 7. Overlap
    overlap_cosine = set(cos_elem["composition"]) & set(cos_comp["composition"])

    # 8. Accuracy
    elem_acc, elem_scores = compute_accuracy(query, list(cos_elem["composition"]))
    comp_acc, comp_scores = compute_accuracy(query, list(cos_comp["composition"]))

    return {
        "query_formula": query,
        "space_correlation": space_corr,
        "element_topk": list(cos_elem["composition"]),
        "composition_topk": list(cos_comp["composition"]),
        "cosine_overlap": list(overlap_cosine),
        "element_accuracy": elem_acc,
        "composition_accuracy": comp_acc,
        "element_scores": elem_scores,
        "composition_scores": comp_scores,
    }




# ============================================================
# 1) Utility: load & align embeddings
# ============================================================
def load_and_align(elem_path, comp_path):
    """
    Loads element & composition embeddings and aligns them
    by composition string, returning aligned arrays and lists.
    """

    E_train_elem, E_test_elem, X_train_elem, X_test_elem = load_embeddings(elem_path)
    E_train_comp, E_test_comp, X_train_comp, X_test_comp = load_embeddings(comp_path)

    # Build dicts keyed by composition string
    train_elem_dict = {str(x): emb for x, emb in zip(X_train_elem, E_train_elem)}
    train_comp_dict = {str(x): emb for x, emb in zip(X_train_comp, E_train_comp)}

    test_elem_dict = {str(x): emb for x, emb in zip(X_test_elem, E_test_elem)}
    test_comp_dict = {str(x): emb for x, emb in zip(X_test_comp, E_test_comp)}

    # Intersections
    common_train = sorted(set(train_elem_dict.keys()) & set(train_comp_dict.keys()))
    common_test = sorted(set(test_elem_dict.keys()) & set(test_comp_dict.keys()))

    print(f"[INFO] Common train samples: {len(common_train)}")
    print(f"[INFO] Common test samples:  {len(common_test)}")

    # Rebuild aligned arrays
    E_train_elem = np.vstack([train_elem_dict[k] for k in common_train])
    E_train_comp = np.vstack([train_comp_dict[k] for k in common_train])
    X_train = common_train

    E_test_elem = np.vstack([test_elem_dict[k] for k in common_test])
    E_test_comp = np.vstack([test_comp_dict[k] for k in common_test])
    X_test = common_test

    return (E_train_elem, E_test_elem, E_train_comp, E_test_comp, X_train, X_test)


# ============================================================
# 2) Evaluate accuracy & overlap over many queries
# ============================================================
def evaluate_retrieval(E_train_elem, E_test_elem,
                       E_train_comp, E_test_comp,
                       X_train, X_test,
                       top_k=5, n_queries=200):
    """
    For n_queries random test samples, compute:
      - chemical accuracy@k for element & composition encoders
      - retrieval overlap@k (cosine-based)
    """

    n_test = len(X_test)
    idxs = np.random.choice(n_test, size=min(n_queries, n_test), replace=False)

    # Normalize once
    E_train_elem_norm = normalize(E_train_elem)
    E_train_comp_norm = normalize(E_train_comp)

    elem_acc_list = []
    comp_acc_list = []
    overlap_list = []

    for i in idxs:
        query = X_test[i]

        q_elem = E_test_elem[i].reshape(1, -1)
        q_comp = E_test_comp[i].reshape(1, -1)

        # Element encoder neighbors
        cos_elem = cosine_search(q_elem, E_train_elem_norm, X_train, top_k)

        # Composition encoder neighbors
        cos_comp = cosine_search(q_comp, E_train_comp_norm, X_train, top_k)

        # Overlap
        set_elem = set(cos_elem["composition"])
        set_comp = set(cos_comp["composition"])
        overlap = len(set_elem & set_comp) / float(top_k)
        overlap_list.append(overlap)

        # Chemical accuracy
        elem_acc, _ = compute_accuracy(query, list(cos_elem["composition"]))
        comp_acc, _ = compute_accuracy(query, list(cos_comp["composition"]))

        elem_acc_list.append(elem_acc)
        comp_acc_list.append(comp_acc)

    return {
        "elem_acc_list": np.array(elem_acc_list),
        "comp_acc_list": np.array(comp_acc_list),
        "overlap_list": np.array(overlap_list),
    }


# ============================================================
# 3) Clustering behaviour (KMeans + AMI)
# ============================================================
def evaluate_clustering(E_train_elem, E_train_comp, n_clusters=20, n_samples=5000):
    """
    Run KMeans on both embedding spaces (same subset) and compute
    adjusted mutual information between cluster assignments.
    """

    n = E_train_elem.shape[0]
    if n > n_samples:
        idxs = np.random.choice(n, size=n_samples, replace=False)
        X_elem = E_train_elem[idxs]
        X_comp = E_train_comp[idxs]
    else:
        X_elem = E_train_elem
        X_comp = E_train_comp

    km_elem = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    km_comp = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")

    labels_elem = km_elem.fit_predict(X_elem)
    labels_comp = km_comp.fit_predict(X_comp)

    ami = adjusted_mutual_info_score(labels_elem, labels_comp)
    return ami, labels_elem, labels_comp


# ============================================================
# 4) t-SNE visualization
# ============================================================
def compute_tsne(E_elem, E_comp, n_samples=2000):
    """
    Compute joint t-SNE coordinates for element & composition embeddings.
    """

    n = E_elem.shape[0]
    if n > n_samples:
        idxs = np.random.choice(n, size=n_samples, replace=False)
        E_elem = E_elem[idxs]
        E_comp = E_comp[idxs]

    # Stack to run joint t-SNE
    stacked = np.vstack([E_elem, E_comp])
    tsne = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto")
    coords = tsne.fit_transform(stacked)

    n_e = E_elem.shape[0]
    elem_coords = coords[:n_e]
    comp_coords = coords[n_e:]

    return elem_coords, comp_coords


# ============================================================
# 5) Hybrid encoder (simple linear combination)
# ============================================================
def build_hybrid_embeddings(E_elem, E_comp, alpha=0.5):
    """
    Simple hybrid: alpha * element + (1-alpha) * composition.
    """
    E_hybrid = alpha * E_elem + (1.0 - alpha) * E_comp
    return normalize(E_hybrid)


# ============================================================
# 6) Create PDF report
# ============================================================
def create_report(elem_path="saved/element_embeddings",
                  comp_path="saved/comp_embeddings",
                  top_k=5,
                  out_pdf="qwen_comparison_report.pdf"):

    # --------------------------------------------------------
    # Load & align
    # --------------------------------------------------------
    (E_train_elem, E_test_elem,
     E_train_comp, E_test_comp,
     X_train, X_test) = load_and_align(elem_path, comp_path)

    # Normalize for global computations
    E_train_elem_norm = normalize(E_train_elem)
    E_train_comp_norm = normalize(E_train_comp)

    # --------------------------------------------------------
    # Space correlation
    # --------------------------------------------------------
    space_corr = compute_space_correlation(E_train_comp_norm, E_train_elem_norm)
    print(f"[INFO] Space correlation: {space_corr:.3f}")

    # --------------------------------------------------------
    # Retrieval & accuracy
    # --------------------------------------------------------
    eval_res = evaluate_retrieval(
        E_train_elem_norm, E_test_elem,
        E_train_comp_norm, E_test_comp,
        X_train, X_test,
        top_k=top_k, n_queries=500  # adjust as you like
    )

    elem_acc = eval_res["elem_acc_list"]
    comp_acc = eval_res["comp_acc_list"]
    overlap = eval_res["overlap_list"]

    print(f"[INFO] Mean element accuracy@{top_k}:      {elem_acc.mean():.3f}")
    print(f"[INFO] Mean composition accuracy@{top_k}: {comp_acc.mean():.3f}")
    print(f"[INFO] Mean overlap@{top_k}:               {overlap.mean():.3f}")

    # --------------------------------------------------------
    # Clustering behaviour
    # --------------------------------------------------------
    ami, labels_elem, labels_comp = evaluate_clustering(
        E_train_elem_norm, E_train_comp_norm,
        n_clusters=20, n_samples=5000
    )
    print(f"[INFO] Clustering AMI (elem vs comp): {ami:.3f}")

    # --------------------------------------------------------
    # t-SNE visualisation
    # --------------------------------------------------------
    elem_coords, comp_coords = compute_tsne(
        E_train_elem_norm, E_train_comp_norm, n_samples=2000
    )

    # --------------------------------------------------------
    # Hybrid embeddings (for reference; only basic metric)
    # --------------------------------------------------------
    E_train_hybrid = build_hybrid_embeddings(E_train_elem_norm, E_train_comp_norm, alpha=0.5)
    hybrid_corr = compute_space_correlation(E_train_hybrid, E_train_elem_norm)
    print(f"[INFO] Hybrid vs element space correlation: {hybrid_corr:.3f}")

    # --------------------------------------------------------
    # BEGIN PDF
    # --------------------------------------------------------
    with PdfPages(out_pdf) as pdf:

        # ------------ Page 1: Summary text ------------
        fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4 ratio
        ax.axis("off")

        text = f"""
Qwen Embedding Comparison Report

Number of aligned train samples: {E_train_elem.shape[0]}
Number of aligned test samples:  {E_test_elem.shape[0]}

Space Correlation (train embeddings):
  element vs composition: {space_corr:.3f}

Mean Chemical Accuracy@{top_k} (over test queries):
  Element encoder:        {elem_acc.mean():.3f}
  Composition encoder:    {comp_acc.mean():.3f}

Mean Retrieval Overlap@{top_k} (cosine):
  Element vs Composition: {overlap.mean():.3f}

Clustering Agreement (KMeans, 20 clusters, AMI):
  AMI(element clustering, composition clustering): {ami:.3f}

Hybrid Embeddings (alpha=0.5 * element + 0.5 * composition):
  Correlation(hybrid, element): {hybrid_corr:.3f}

Notes:
- Element encoder uses chemically-weighted element embeddings.
- Composition encoder uses Qwen directly on formula strings.
- Low space correlation indicates very different geometry.
- High chemical accuracy for both suggests both retrieve
  chemically related compounds, but with different neighborhoods.
"""

        ax.text(0.05, 0.95, text, va="top", ha="left", fontsize=10, family="monospace")
        pdf.savefig(fig)
        plt.close(fig)

        # ------------ Page 2: Accuracy & overlap histograms ------------
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].hist(elem_acc, bins=10)
        axes[0].set_title(f"Element accuracy@{top_k}")
        axes[0].set_xlabel("accuracy")
        axes[0].set_ylabel("#queries")

        axes[1].hist(comp_acc, bins=10)
        axes[1].set_title(f"Composition accuracy@{top_k}")
        axes[1].set_xlabel("accuracy")

        axes[2].hist(overlap, bins=10)
        axes[2].set_title(f"Overlap@{top_k}")
        axes[2].set_xlabel("fraction overlap")

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ------------ Page 3: t-SNE scatter ------------
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(elem_coords[:, 0], elem_coords[:, 1], s=5, alpha=0.5, label="Element space")
        ax.scatter(comp_coords[:, 0], comp_coords[:, 1], s=5, alpha=0.5, label="Composition space")
        ax.set_title("t-SNE of embeddings (train subset)")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend()
        pdf.savefig(fig)
        plt.close(fig)

        # ------------ Page 4: Clustering histograms ------------
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].hist(labels_elem, bins=20)
        axes[0].set_title("Cluster distribution (element)")

        axes[1].hist(labels_comp, bins=20)
        axes[1].set_title("Cluster distribution (composition)")

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
            # ------------ Accuracy bar chart ------------
        fig, ax = plt.subplots(figsize=(5, 4))

        means = [elem_acc.mean(), comp_acc.mean()]
        stds = [elem_acc.std(), comp_acc.std()]
        labels = ["Element encoder", "Composition encoder"]

        ax.bar(labels, means, yerr=stds, capsize=6)
        ax.set_ylabel(f"Chemical accuracy@{top_k}")
        ax.set_ylim(0, 1)
        ax.set_title("Retrieval accuracy comparison")

        for i, v in enumerate(means):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center")

        pdf.savefig(fig)
        plt.close(fig)


    print(f"[DONE] Report written to: {out_pdf}")


if __name__ == "__main__":
    create_report(
        elem_path="saved/element_embeddings",
        comp_path="saved/comp_embeddings",
        top_k=5,
        out_pdf="qwen_comparison_report.pdf",
    )