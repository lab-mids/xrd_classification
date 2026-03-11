import numpy as np
from numpy.linalg import norm
from collections import Counter, defaultdict
import math
import matplotlib.pyplot as plt
from collections import defaultdict
from collections import defaultdict
import numpy as np
from numpy.linalg import norm
def robust_latent_classifier(
    saved_latents,          # (N, D)
    saved_labels,           # (N,)
    class_centroids,        # dict[label -> (D,)]
    sample_latent,          # (D,)
    k_schedule=(50, 100, 200, 500),
    radius_schedule=(0.4, 0.6, 0.8, 1.0),
    confidence_threshold=0.35,
    entropy_threshold=0.65,
    gap_threshold=0.12,
    proto_agreement_threshold=0.1,
    min_avg_cosine=0.5,
    eps=1e-8
):
    """
    Robust latent classifier (NON-AMBIGUOUS).

    - Always returns a predicted label
    - Prototype disagreement reduces confidence instead of blocking
    """

    # --------------------------------------------------
    # 0. Validation
    # --------------------------------------------------
    if not isinstance(class_centroids, dict) or len(class_centroids) == 0:
        raise ValueError("class_centroids must be a non-empty dict")

    # --------------------------------------------------
    # 1. Normalize
    # --------------------------------------------------
    train_norm = saved_latents / (norm(saved_latents, axis=1, keepdims=True) + eps)
    q = sample_latent / (norm(sample_latent) + eps)

    cosine_scores = train_norm @ q
    euclidean_dist = norm(train_norm - q, axis=1)

    unique_classes = np.unique(saved_labels)
    num_classes = len(unique_classes)

    # --------------------------------------------------
    # 2. Prototype similarity (global evidence)
    # --------------------------------------------------
    proto_scores = {}
    for lbl, proto in class_centroids.items():
        proto_norm = proto / (norm(proto) + eps)
        proto_scores[lbl] = float(proto_norm @ q)

    proto_sorted = sorted(proto_scores.items(), key=lambda x: x[1], reverse=True)
    proto_label, proto_score = proto_sorted[0]
    proto_gap = (
        proto_score - proto_sorted[1][1]
        if len(proto_sorted) > 1 else 1.0
    )

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------
    def shannon_entropy(probs):
        return -sum(p * math.log(p + eps) for p in probs)

    def normalize(counter):
        s = sum(counter.values())
        return {k: v / s for k, v in counter.items()}

    sorted_idx = np.argsort(cosine_scores)[::-1]

    best_result = None
    best_score = -np.inf

    # --------------------------------------------------
    # 3. Adaptive kNN
    # --------------------------------------------------
    for k in k_schedule:
        idx_k = sorted_idx[:k]

        votes = defaultdict(float)
        sims = []

        for idx in idx_k:
            lbl = saved_labels[idx]
            sim = float(cosine_scores[idx])
            votes[lbl] += sim
            sims.append(sim)

        if not sims:
            continue

        avg_cos = float(np.mean(sims))
        if avg_cos < min_avg_cosine:
            break

        probs = normalize(votes)
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)

        knn_label, knn_prob = sorted_probs[0]
        second_prob = sorted_probs[1][1] if len(sorted_probs) > 1 else 0.0

        confidence = knn_prob
        entropy = shannon_entropy(probs.values()) / math.log(num_classes + eps)
        gap = knn_prob - second_prob

        # --------------------------------------------------
        # 4. Radius stability
        # --------------------------------------------------
        stable = False
        for r in radius_schedule:
            idx_inside = np.where(euclidean_dist <= r)[0]
            if len(idx_inside) == 0:
                continue

            weights = defaultdict(float)
            for idx in idx_inside:
                lbl = saved_labels[idx]
                weights[lbl] += 1.0 / (euclidean_dist[idx] + eps)

            if max(weights, key=weights.get) == knn_label:
                stable = True
                break

        # --------------------------------------------------
        # 5. SOFT prototype penalty (core fix)
        # --------------------------------------------------
        proto_penalty = 1.0

        if knn_label != proto_label:
            proto_penalty *= 0.7
        elif proto_gap < proto_agreement_threshold:
            proto_penalty *= proto_gap / proto_agreement_threshold

        final_confidence = confidence * proto_penalty

        score = final_confidence * gap

        candidate = {
            "predicted_label": knn_label,
            "status": "confident" if (
                final_confidence >= confidence_threshold and
                entropy <= entropy_threshold and
                gap >= gap_threshold and
                stable
            ) else "low_confidence",
            "k_used": k,
            "confidence": round(final_confidence, 3),
            "raw_confidence": round(confidence, 3),
            "entropy": round(entropy, 3),
            "gap": round(gap, 3),
            "avg_cosine_similarity": round(avg_cos, 4),
            "prototype_label": proto_label,
            "prototype_score": round(proto_score, 4),
            "prototype_gap": round(proto_gap, 4),
            "class_probabilities": {
                c: round(p, 3) for c, p in probs.items()
            }
        }

        if score > best_score:
            best_score = score
            best_result = candidate

    # --------------------------------------------------
    # 6. Guaranteed return (never ambiguous)
    # --------------------------------------------------
    return best_result

# ================================
# 4️⃣ Adaptive Retrieval Function
# ================================

def adaptive_cosine_retrieval_classwise(
    saved_latents_norm,
    saved_labels,
    sample_latent,
    k_range=(50, 100, 200, 300),
    min_avg_cosine=0.5,
    min_confidence=0.3,
    min_margin=0.1
):
    import numpy as np

    cosine_scores = saved_latents_norm @ sample_latent

    max_k = min(max(k_range), len(cosine_scores))

    top_idx = np.argpartition(-cosine_scores, max_k - 1)[:max_k]
    sorted_idx = top_idx[np.argsort(-cosine_scores[top_idx])]

    best_result = None
    best_soft_score = -np.inf
    best_soft_result = None

    for k in k_range:
        if k > len(sorted_idx):
            continue

        idx_k = sorted_idx[:k]
        similarities = cosine_scores[idx_k]
        labels_k = saved_labels[idx_k]

        avg_cos = float(np.mean(similarities))
        if avg_cos < min_avg_cosine:
            break

        unique_labels, inverse = np.unique(labels_k, return_inverse=True)
        vote_sums = np.zeros(len(unique_labels))

        np.add.at(vote_sums, inverse, similarities)

        sorted_order = np.argsort(-vote_sums)
        vote_sums = vote_sums[sorted_order]
        unique_labels = unique_labels[sorted_order]

        pred_label = unique_labels[0]
        total_votes = np.sum(vote_sums)

        if len(vote_sums) == 1:
            margin = 1.0
        else:
            margin = (vote_sums[0] - vote_sums[1]) / total_votes

        confidence = vote_sums[0] / total_votes
        soft_score = confidence * margin

        candidate = {
            "predicted_label": pred_label,
            "confidence": float(confidence),
            "margin": float(margin),
            "selected_top_k": k
        }

        if confidence >= min_confidence and margin >= min_margin:
            best_result = candidate

        if soft_score > best_soft_score:
            best_soft_score = soft_score
            best_soft_result = candidate

    if best_result is not None:
        return best_result

    if best_soft_result is not None:
        return best_soft_result

    # Fallback: nearest neighbor
    top_idx = sorted_idx[0]
    return {
        "predicted_label": saved_labels[top_idx],
        "confidence": 1.0,
        "margin": 1.0,
        "selected_top_k": 1
    }
def adaptive_cosine_retrieval_classwise_topN(
    saved_latents,
    saved_labels,
    sample_latent,
    k_range=(50, 100, 200, 300, 500, 800, 1000),
    top_n=3,
    min_avg_cosine=0.5
):
    from collections import defaultdict
    import numpy as np
    from numpy.linalg import norm

    # --------------------------------------------------
    # 1. Normalize embeddings
    # --------------------------------------------------
    train_norm = saved_latents      # already normalized
    sample_norm = sample_latent     # already normalized
    # --------------------------------------------------
    # 2. Cosine similarity
    # --------------------------------------------------
    cosine_scores = train_norm @ sample_norm
    sorted_idx = np.argsort(cosine_scores)[::-1]

    # Store best score per class across all k
    class_best = {}

    # --------------------------------------------------
    # 3. Adaptive k loop
    # --------------------------------------------------
    for k in k_range:
        idx_k = sorted_idx[:k]

        votes = defaultdict(float)
        similarities = []

        for idx in idx_k:
            lbl = saved_labels[idx]
            sim = float(cosine_scores[idx])
            votes[lbl] += sim
            similarities.append(sim)

        if not similarities:
            continue

        avg_cos = float(np.mean(similarities))
        if avg_cos < min_avg_cosine:
            break

        total_vote = sum(votes.values())

        # --------------------------------------------------
        # 4. Compute per-class score
        # --------------------------------------------------
        sorted_votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)

        for i, (label, vote) in enumerate(sorted_votes):
            confidence = vote / total_vote

            if i == 0 and len(sorted_votes) > 1:
                margin = (vote - sorted_votes[1][1]) / total_vote
            else:
                margin = confidence

            soft_score = confidence * margin

            candidate = {
            "label": int(label),
            "prob": float(soft_score),   
            "confidence": round(confidence, 3),
            "margin": round(margin, 3),
            "soft_score": round(soft_score, 4),
            "avg_cosine_similarity": round(avg_cos, 4),
            "k": k,}

            # keep best version of this class
            if label not in class_best or soft_score > class_best[label]["soft_score"]:
                class_best[label] = candidate

    # --------------------------------------------------
    # 5. Rank classes globally
    # --------------------------------------------------
    ranked = sorted(
        class_best.values(),
        key=lambda x: x["soft_score"],
        reverse=True
    )

    return {
        "predicted_label": ranked[0]["label"] if ranked else None,
        "top_candidates": ranked[:top_n]
    }
def adaptive_cosine_fast(
    train_norm,
    saved_labels,
    sample_latent,
    k_range=(50,100,200,300,500,800,1000),
    top_n=1,
    min_avg_cosine=-1
):
    from collections import defaultdict
    import numpy as np

    eps = 1e-8
    sample_norm = sample_latent / (np.linalg.norm(sample_latent) + eps)

    cosine_scores = train_norm @ sample_norm

    max_k = max(k_range)
    idx_partial = np.argpartition(cosine_scores, -max_k)[-max_k:]
    idx_sorted = idx_partial[np.argsort(cosine_scores[idx_partial])[::-1]]

    class_best = {}

    for k in k_range:
        idx_k = idx_sorted[:k]

        votes = defaultdict(float)
        similarities = cosine_scores[idx_k]

        avg_cos = float(np.mean(similarities))
        if avg_cos < min_avg_cosine:
            break

        for idx in idx_k:
            lbl = saved_labels[idx]
            votes[lbl] += float(cosine_scores[idx])

        total_vote = sum(votes.values())

        sorted_votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)

        for i, (label, vote) in enumerate(sorted_votes):
            confidence = vote / total_vote

            if i == 0 and len(sorted_votes) > 1:
                margin = (vote - sorted_votes[1][1]) / total_vote
            else:
                margin = confidence

            soft_score = confidence * margin

            if label not in class_best or soft_score > class_best[label]["soft_score"]:
                class_best[label] = {
                    "label": label,
                    "soft_score": soft_score
                }

    ranked = sorted(class_best.values(), key=lambda x: x["soft_score"], reverse=True)

    return ranked[:top_n]

def retrieve_by_euclidean_radius_classwise(
    saved_latents,      # (N, D)
    saved_labels,       # (N,)
    sample_latent,      # (D,)
    radius=0.8
):
    """
    Euclidean-radius classwise retrieval.

    Steps:
    - Normalize embeddings
    - Select all points inside a radius
    - Compute per-class:
        * count
        * average distance
    - Select best class by:
        1) maximum count
        2) minimum average distance (tie-break)
    """

    # --------------------------------------------------
    # 1. Normalize embeddings
    # --------------------------------------------------
    train_norm = saved_latents / norm(saved_latents, axis=1, keepdims=True)
    sample_norm = sample_latent / norm(sample_latent)

    # --------------------------------------------------
    # 2. Euclidean distances
    # --------------------------------------------------
    euclidean_dist = norm(train_norm - sample_norm, axis=1)

    # --------------------------------------------------
    # 3. Points inside the radius
    # --------------------------------------------------
    idx_inside = np.where(euclidean_dist <= radius)[0]

    if len(idx_inside) == 0:
        return {
            "radius": radius,
            "num_neighbors": 0,
            "best_class": None,
            "class_stats": {},
            "global_avg_distance": None,
        }

    # --------------------------------------------------
    # 4. Collect per-class distances
    # --------------------------------------------------
    class_distances = defaultdict(list)

    for idx in idx_inside:
        label = saved_labels[idx]
        class_distances[label].append(float(euclidean_dist[idx]))

    # --------------------------------------------------
    # 5. Per-class statistics
    # --------------------------------------------------
    class_stats = {
        label: {
            "count": len(dists),
            "avg_distance": float(np.mean(dists))
        }
        for label, dists in class_distances.items()
    }

    # --------------------------------------------------
    # 6. Select best class
    # --------------------------------------------------
    best_class = min(
        class_stats.items(),
        key=lambda x: (-x[1]["count"], x[1]["avg_distance"])
    )[0]

    all_distances = [d for dists in class_distances.values() for d in dists]

    return {
        "radius": radius,
        "num_neighbors": len(all_distances),
        "best_class": best_class,
        "class_stats": class_stats,
        "global_avg_distance": float(np.mean(all_distances)),
    }


def auto_radius_and_stability(
    saved_latents,
    saved_labels,
    sample_latent,
    radius_min=0.6,
    radius_max=1.2,
    num_steps=7,
    min_neighbors=20
):
    """
    Automatically select a stable radius and dominant class.

    Procedure:
    - Sweep radii
    - Keep radii with enough neighbors
    - Compute class stability across radii
    - Select:
        * dominant class (most frequent)
        * smallest radius where it appears
    """

    radii = np.linspace(radius_min, radius_max, num_steps)

    per_radius_results = []
    valid_results = []

    # --------------------------------------------------
    # 1. Sweep radii
    # --------------------------------------------------
    for r in radii:
        out = retrieve_by_euclidean_radius_classwise(
            saved_latents=saved_latents,
            saved_labels=saved_labels,
            sample_latent=sample_latent,
            radius=float(r)
        )

        entry = {
            "radius": float(r),
            "best_class": out["best_class"],
            "num_neighbors": out["num_neighbors"],
            "class_stats": out["class_stats"],
        }

        per_radius_results.append(entry)

        if (
            out["best_class"] is not None
            and out["num_neighbors"] >= min_neighbors
        ):
            valid_results.append(entry)

    # --------------------------------------------------
    # 2. No valid radii → reject
    # --------------------------------------------------
    if len(valid_results) == 0:
        return {
            "selected_radius": None,
            "best_class": None,
            "stability_scores": {},
            "per_radius": per_radius_results,
        }

    # --------------------------------------------------
    # 3. Stability scores
    # --------------------------------------------------
    class_occurrences = Counter(v["best_class"] for v in valid_results)
    total_valid = len(valid_results)

    stability_scores = {
        cls: count / total_valid
        for cls, count in class_occurrences.items()
    }

    # --------------------------------------------------
    # 4. Dominant class
    # --------------------------------------------------
    best_class = max(stability_scores.items(), key=lambda x: x[1])[0]

    # --------------------------------------------------
    # 5. Smallest radius where it appears
    # --------------------------------------------------
    selected_radius = next(
        v["radius"] for v in valid_results
        if v["best_class"] == best_class
    )

    return {
        "selected_radius": selected_radius,
        "best_class": best_class,
        "stability_scores": stability_scores,
        "per_radius": per_radius_results,
    }

def predict_label_radius_weighted(
    saved_latents,
    saved_indices,
    linked_vectors,
    sample_latent,
    radius=1.0,
    eps=1e-8
):
    # Normalize
    train_norm = saved_latents / norm(saved_latents, axis=1, keepdims=True)
    q = sample_latent / norm(sample_latent)

    # Distances
    d = norm(train_norm - q, axis=1)

    # Neighbors within radius
    idx_inside = np.where(d <= radius)[0]
    if len(idx_inside) == 0:
        return {"predicted_label": None, "probabilities": {}, "num_neighbors": 0}

    linked_dict = {e["sample_index"]: e for e in linked_vectors}

    # Distance-weighted voting
    weights = defaultdict(float)
    for idx in idx_inside:
        gidx = int(saved_indices[idx])
        label = linked_dict[gidx].get("shap_label")
        if label is None:
            continue
        w = 1.0 / (float(d[idx]) + eps)   # closer => bigger weight
        weights[label] += w

    if not weights:
        return {"predicted_label": None, "probabilities": {}, "num_neighbors": 0}

    total = sum(weights.values())
    probs = {k: v / total for k, v in weights.items()}
    pred = max(probs, key=probs.get)

    return {
        "predicted_label": pred,
        "probabilities": probs,
        "num_neighbors": int(len(idx_inside))
    }


def retrieve_by_cosine_similarity(
    saved_latents,   # (N, D) -> z_cls
    saved_labels,    # (N,)
    sample_latent,   # (D,)   -> z_cls
    top_k=100
):
    """
    Simple TOP-K cosine similarity retrieval on z_cls.
    - Majority vote (unweighted)
    """

    # --------------------------------------------------
    # 1. Normalize embeddings
    # --------------------------------------------------
    train_norm = saved_latents / norm(saved_latents, axis=1, keepdims=True)
    sample_norm = sample_latent / norm(sample_latent)

    # --------------------------------------------------
    # 2. Cosine similarity
    # --------------------------------------------------
    cosine_scores = train_norm @ sample_norm
    idx_sorted = np.argsort(cosine_scores)[::-1][:top_k]

    # --------------------------------------------------
    # 3. Collect labels
    # --------------------------------------------------
    labels = saved_labels[idx_sorted]
    similarities = cosine_scores[idx_sorted]

    if len(labels) == 0:
        return {
            "method": "cosine_similarity_classwise",
            "top_k": top_k,
            "predicted_label": "ambiguous",
            "confidence": 0.0,
            "num_neighbors": 0,
            "avg_cosine_similarity": None,
            "class_counts": {}
        }

    # --------------------------------------------------
    # 4. Majority vote
    # --------------------------------------------------
    class_counts = Counter(labels)
    pred_label, count = class_counts.most_common(1)[0]
    confidence = count / len(labels)

    return {
        "method": "cosine_similarity_classwise",
        "top_k": top_k,
        "predicted_label": pred_label,
        "confidence": round(confidence, 3),
        "num_neighbors": len(labels),
        "avg_cosine_similarity": round(float(np.mean(similarities)), 4),
        "class_counts": dict(class_counts)
    }


def retrieve_from_saved_latents_cosine(
    saved_latents,      # (N, D) -> z_cls
    saved_labels,       # (N,)
    sample_latent,      # (D,)   -> z_cls
    top_k=100,
    euclidean_radius=None
):
    """
    Nearest-neighbor retrieval using cosine similarity on z_cls.

    Returns:
      - predicted label (majority vote)
      - confidence
      - best overall match
      - best match within predicted class
      - class distribution
    """

    # --------------------------------------------------
    # 1. Normalize embeddings
    # --------------------------------------------------
    train_norm = saved_latents / norm(saved_latents, axis=1, keepdims=True)
    sample_norm = sample_latent / norm(sample_latent)

    # --------------------------------------------------
    # 2. Similarity & distance
    # --------------------------------------------------
    cosine_scores = train_norm @ sample_norm
    euclidean_dist = norm(train_norm - sample_norm, axis=1)

    # --------------------------------------------------
    # 3. Optional radius filtering
    # --------------------------------------------------
    if euclidean_radius is not None:
        idx_candidates = np.where(euclidean_dist <= euclidean_radius)[0]
        if len(idx_candidates) == 0:
            idx_candidates = np.arange(len(cosine_scores))
    else:
        idx_candidates = np.arange(len(cosine_scores))

    # --------------------------------------------------
    # 4. Sort by cosine similarity & select top-k
    # --------------------------------------------------
    idx_sorted = idx_candidates[
        np.argsort(cosine_scores[idx_candidates])[::-1]
    ][:top_k]

    if len(idx_sorted) == 0:
        return {
            "predicted_label": "ambiguous",
            "confidence": 0.0,
            "num_neighbors": 0,
            "best_match": None,
            "best_predicted_class_match": None,
            "class_counts": {}
        }

    # --------------------------------------------------
    # 5. Best overall match (rank-1)
    # --------------------------------------------------
    best_idx = idx_sorted[0]

    best_match = {
        "label": saved_labels[best_idx],
        "cosine_similarity": float(cosine_scores[best_idx]),
        "euclidean_distance": float(euclidean_dist[best_idx]),
        "index": int(best_idx)
    }

    # --------------------------------------------------
    # 6. Collect labels & best per class
    # --------------------------------------------------
    collected_labels = []
    best_per_class = {}

    for idx in idx_sorted:
        label = saved_labels[idx]
        collected_labels.append(label)

        # First occurrence = highest cosine (sorted order)
        if label not in best_per_class:
            best_per_class[label] = {
                "label": label,
                "cosine_similarity": float(cosine_scores[idx]),
                "euclidean_distance": float(euclidean_dist[idx]),
                "index": int(idx)
            }

    # --------------------------------------------------
    # 7. Voting
    # --------------------------------------------------
    class_counts = Counter(collected_labels)
    most_common_label, most_common_count = class_counts.most_common(1)[0]
    confidence = most_common_count / len(collected_labels)

    best_predicted_class_match = best_per_class.get(most_common_label)

    # --------------------------------------------------
    # 8. Final output
    # --------------------------------------------------
    return {
        "predicted_label": most_common_label,
        "confidence": round(confidence, 3),
        "num_neighbors": len(collected_labels),
        "best_match": best_match,
        "best_predicted_class_match": best_predicted_class_match,
        "class_counts": dict(class_counts)
    }

def plot_class_counts_vs_radius(auto_out, save_path):
    """
    Plot number of samples per class vs radius and save figure.
    """

    per_radius = auto_out["per_radius"]
    radii = [entry["radius"] for entry in per_radius]

    # Collect all class names
    all_classes = set()
    for entry in per_radius:
        all_classes.update(entry["class_stats"].keys())

    # Initialize counts
    class_counts = {cls: [] for cls in all_classes}

    # Fill counts
    for entry in per_radius:
        stats = entry["class_stats"]
        for cls in all_classes:
            class_counts[cls].append(
                stats.get(cls, {}).get("count", 0)
            )

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    plt.figure(figsize=(9, 6))

    for cls, counts in class_counts.items():
        plt.plot(radii, counts, marker="o", label=cls)

    selected_r = auto_out.get("selected_radius")
    if selected_r is not None:
        plt.axvline(
            selected_r,
            color="k",
            linestyle="--",
            alpha=0.6,
            label="Selected radius"
        )

    plt.xlabel("Radius")
    plt.ylabel("Number of samples")
    plt.title("Class counts vs radius")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()   


def cosine_similarity(a, b, eps=1e-8):
    a = a / (norm(a) + eps)
    b = b / (norm(b) + eps)
    return float(a @ b)
