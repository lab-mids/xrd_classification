
import numpy as np
from collections import Counter
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
from numpy.linalg import norm
from collections import Counter

def extract_latent_vectors(model, cs, device, loader_name="train_loader"):
    data_loader = cs[loader_name]
    model.eval()

    latents = []
    labels = []

    with torch.no_grad():
        for xb, yb in data_loader:
            xb = xb.to(device).float()

            # encoder returns BOTH
            z_rec, z_cls = model.encode(xb)

            #  USE CLASSIFICATION LATENT
            latents.append(z_cls.cpu().numpy())
            labels.append(yb.numpy())

    latents = np.concatenate(latents, axis=0)
    labels  = np.concatenate(labels, axis=0)
    indices = np.arange(len(latents))

    return latents, labels, indices


def extract_latent_vectors_from_loader(
    model,
    dataloader,
    device,
    latent_type="cls",   # "cls" or "rec"
):
    model.eval()

    latents = []
    labels = []

    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device).float()

            z_rec, z_cls = model.encode(xb)

            # choose SAME latent space
            if latent_type == "cls":
                z = z_cls
            else:
                z = z_rec

            latents.append(z.cpu().numpy())
            labels.append(yb.numpy())

    latents = np.concatenate(latents, axis=0)
    labels  = np.concatenate(labels, axis=0)
    indices = np.arange(len(latents))

    return latents, labels, indices


def retrieve_from_saved_latents_umap(
        saved_latents,
        saved_indices,
        linked_vectors,
        sample_latent,
        all_labels,          # array of label IDs for all training samples
        class_names,         # class_id → class_name
        top_k=10,
        n_neighbors=50,
        min_dist=0.1,
        n_components=2,
        percent_local=1      # percent of nearest UMAP points for local dist
    ):
    """
    UMAP retrieval using saved training latents + queried sample latent.

    Returns:
    - Top-k neighbors
    - Class distribution among top-k
    - Local UMAP class distribution (closest X%)
    - UMAP embedding for training and sample
    """

    # -----------------------------------
    # 1. Fit UMAP on training latents
    # -----------------------------------
    umap_model = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric="euclidean",
        random_state=42
    )

    train_umap = umap_model.fit_transform(saved_latents)

    # -----------------------------------
    # 2. Transform the query sample
    # -----------------------------------
    sample_umap = umap_model.transform(sample_latent.reshape(1, -1))[0]

    # -----------------------------------
    # 3. Compute distances (UMAP space)
    # -----------------------------------
    dists = np.linalg.norm(train_umap - sample_umap, axis=1)
    idx_sorted = np.argsort(dists)[:top_k]

    # build lookup table
    linked_dict = {entry["sample_index"]: entry for entry in linked_vectors}

    results = []
    collected_labels = []

    # -----------------------------------
    # 4. Build results list
    # -----------------------------------
    for rank, idx in enumerate(idx_sorted):
        gidx = int(saved_indices[idx])
        base_entry = linked_dict[gidx].copy()

        label_name = base_entry.get("label_name", None)
        if label_name is not None:
            collected_labels.append(label_name)

        results.append({
            "rank": rank + 1,
            "umap_distance": float(dists[idx]),
            "sample_index": gidx,
            "shap_peaks": base_entry["shap_peaks"],
            "embedding": base_entry["embedding"],
            "label_name": label_name,
            "label_id": base_entry.get("label_id", None),
        })

    # -----------------------------------
    # 5. Top-k class distribution
    # -----------------------------------
    class_counts = Counter(collected_labels)

    most_common_label, most_common_count = (None, 0)
    if class_counts:
        most_common_label, most_common_count = class_counts.most_common(1)[0]

    # -----------------------------------
    # 6. Local UMAP class distribution (closest X%)
    # -----------------------------------
    threshold = np.percentile(dists, percent_local)
    mask_local = dists < threshold
    local_labels = all_labels[mask_local]

    local_counts = Counter(local_labels)

    local_class_distribution = {
        class_names[cid]: count for cid, count in local_counts.items()
    }

    # -----------------------------------
    # 7. Return everything
    # -----------------------------------
    return {
        "neighbors": results,
        "class_counts": dict(class_counts),
        "most_common_label": most_common_label,
        "most_common_count": most_common_count,
        "train_umap": train_umap,
        "sample_umap": sample_umap,
        "local_umap_distribution": local_class_distribution,
        "local_percent_used": percent_local,
        "local_n_points": int(mask_local.sum())
    }


def retrieve_from_saved_latents_tsne(
        saved_latents,
        saved_indices,
        linked_vectors,
        sample_latent,
        all_labels,          # NEW: array of class IDs for all training samples
        class_names,         # NEW: mapping: class_id → class_name
        top_k=10,
        perplexity=40,
        percent_local=1      # % of closest t-SNE points to include in local dist
    ):
    """
    t-SNE retrieval including:
    - Top-k neighbors
    - Class distribution among neighbors
    - Local t-SNE class distribution (closest X%)
    """

    # -------------------------------
    # 1. PCA reduction
    # -------------------------------
    pca = PCA(n_components=50)
    train_pca  = pca.fit_transform(saved_latents)
    sample_pca = pca.transform(sample_latent.reshape(1, -1))

    # -------------------------------
    # 2. Joint t-SNE
    # -------------------------------
    tsne_input = np.vstack([train_pca, sample_pca])
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=200,
        n_iter=1500,
        verbose=0
    )
    tsne_out = tsne.fit_transform(tsne_input)

    train_tsne  = tsne_out[:-1]
    sample_tsne = tsne_out[-1]

    # -------------------------------
    # 3. Compute distances for top-k
    # -------------------------------
    dists = np.linalg.norm(train_tsne - sample_tsne, axis=1)
    idx_sorted = np.argsort(dists)[:top_k]

    # convert list → dict for fast lookup
    linked_dict = {entry["sample_index"]: entry for entry in linked_vectors}

    results = []
    collected_labels = []

    # -------------------------------
    # 4. Build results list
    # -------------------------------
    for rank, idx in enumerate(idx_sorted):

        gidx = int(saved_indices[idx])
        base_entry = linked_dict[gidx].copy()

        label_name = base_entry.get("label_name", None)
        if label_name is not None:
            collected_labels.append(label_name)

        results.append({
            "rank": rank + 1,
            "tsne_distance": float(dists[idx]),
            "sample_index": gidx,
            "shap_peaks": base_entry["shap_peaks"],
            "embedding": base_entry["embedding"],
            "label_name": label_name,
            "label_id": base_entry.get("label_id", None),
        })

    # -------------------------------
    # 5. Global class counts (top-k)
    # -------------------------------
    class_counts = Counter(collected_labels)

    most_common_label, most_common_count = (None, 0)
    if class_counts:
        most_common_label, most_common_count = class_counts.most_common(1)[0]

    # -------------------------------
    # 6. LOCAL t-SNE class distribution (closest X%)
    # -------------------------------
    sample_point = sample_tsne.reshape(1, 2)
    tsne_dists = np.linalg.norm(train_tsne - sample_point, axis=1)

    # threshold for "local neighborhood"
    threshold = np.percentile(tsne_dists, percent_local)

    mask_tsne = tsne_dists < threshold
    local_labels = all_labels[mask_tsne]

    local_counts = Counter(local_labels)

    # convert class IDs → names (like your printed output)
    local_class_distribution = {
        class_names[cls_id]: count for cls_id, count in local_counts.items()
    }

    # -------------------------------
    # 7. Return enriched output
    # -------------------------------
    return {
        "neighbors": results,
        "class_counts": dict(class_counts),
        "most_common_label": most_common_label,
        "most_common_count": most_common_count,
        "train_tsne": train_tsne,
        "sample_tsne": sample_tsne,
        "local_tsne_distribution": local_class_distribution,  
        "local_percent_used": percent_local,                  
        "local_n_points": int(mask_tsne.sum())                
    }




def retrieve_from_saved_latents_cosine(
        saved_latents,      # (N, D) training embeddings
        saved_indices,      # (N,) global sample indices
        linked_vectors,     # list of dictionaries with metadata
        sample_latent,      # (D,) query embedding
        top_k=10
    ):
    """
    Nearest-neighbor retrieval using COSINE SIMILARITY,
    using 'shap_label' as the class label.
    """

    # -------------------------------
    # 1. Normalize embeddings
    # -------------------------------
    train_norm = saved_latents / norm(saved_latents, axis=1, keepdims=True)
    sample_norm = sample_latent / norm(sample_latent)

    # -------------------------------
    # 2. Cosine similarity
    # -------------------------------
    scores = train_norm @ sample_norm  # dot product

    idx_sorted = np.argsort(scores)[::-1][:top_k]

    # Lookup table for metadata
    linked_dict = {entry["sample_index"]: entry for entry in linked_vectors}

    results = []
    collected_labels = []

    # -------------------------------
    # 3. Build result entries
    # -------------------------------
    for rank, idx in enumerate(idx_sorted):

        gidx = int(saved_indices[idx])
        base_entry = linked_dict[gidx].copy()

        # USE SHAP LABEL (this is your actual label)
        label_name = base_entry.get("shap_label", None)

        if label_name is not None:
            collected_labels.append(label_name)

        results.append({
            "rank": rank + 1,
            "cosine_similarity": float(scores[idx]),
            "sample_index": gidx,
            "shap_peaks": base_entry["shap_peaks"],
            "embedding": base_entry["embedding"],
            "shap_label": label_name,   # renamed correctly
        })

    # -------------------------------
    # 4. Class distribution
    # -------------------------------
    class_counts = Counter(collected_labels)

    most_common_label, most_common_count = (None, 0)
    if class_counts:
        most_common_label, most_common_count = class_counts.most_common(1)[0]

    # -------------------------------
    # 5. Return result dictionary
    # -------------------------------
    return {
        #"neighbors": results,
        "class_counts": dict(class_counts),
        "most_common_label": most_common_label,
        "most_common_count": most_common_count,
        "cosine_scores": scores
    }



def cosine_search_composition(
        query_vector,
        saved_embeddings,
        saved_indices,
        saved_labels,
        linked_vectors,
        top_k=10
    ):
    """
    Cosine similarity retrieval for composition embeddings.
    """

    # Normalize
    saved_norm = saved_embeddings / norm(saved_embeddings, axis=1, keepdims=True)
    query_norm = query_vector / norm(query_vector)

    # Cosine similarity
    scores = saved_norm @ query_norm

    # Get top-k
    idx_sorted = np.argsort(scores)[::-1][:top_k]

    results = []
    for rank, idx in enumerate(idx_sorted):
        entry = linked_vectors[idx]

        results.append({
            "rank": rank + 1,
            "sample_index": int(saved_indices[idx]),
            "cosine_similarity": float(scores[idx]),
            "shap_label": saved_labels[idx],
            "shap_peaks": entry["shap_peaks"],
            "embedding": entry["embedding"],
        })

    # Class distribution
    class_counts = Counter(saved_labels[idx_sorted])
    best_label, best_count = class_counts.most_common(1)[0]

    return {
        "top_k_results": results,
        "class_distribution": dict(class_counts),
        "most_common_label": best_label,
        "most_common_count": best_count,
        "scores": scores,
    }

def retrieve_composition_similarity(composition, encoder, linked_vectors, top_k=10):
    """
    1. Encode user composition to embedding
    2. Compare with saved composition embeddings in linked_vectors
    """

    # Step 1 — Encode the composition using Qwen encoder
    query_vec = encoder.encode([composition])[0]

    # Step 2 — Extract saved embeddings and metadata
    saved_comp_embeddings = np.array([e["embedding"] for e in linked_vectors], dtype=float)
    saved_comp_indices = np.array([e["sample_index"] for e in linked_vectors], dtype=int)
    saved_comp_labels = np.array([e["shap_label"] for e in linked_vectors])

    # Step 3 — Run cosine similarity search
    return cosine_search_composition(
        query_vector=query_vec,
        saved_embeddings=saved_comp_embeddings,
        saved_indices=saved_comp_indices,
        saved_labels=saved_comp_labels,
        linked_vectors=linked_vectors,
        top_k=top_k
    )
