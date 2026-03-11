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

    #print(f"[INFO] Encoding {len(compositions)} compositions...")

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

    #print(f"[INFO] Encoding {len(compositions)} compositions...")

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
