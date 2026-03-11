from captum.attr import IntegratedGradients
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import h5py
from scipy.cluster.hierarchy import linkage, dendrogram
from captum.attr import IntegratedGradients
import numpy as np
from scipy.ndimage import gaussian_filter
def plot_ig_only(importance, true_class=None, pred_class=None):
    plt.figure(figsize=(14, 4))
    plt.plot(importance, linewidth=1.5)
    
    title = "Integrated Gradients Importance"
    if true_class is not None:
        title += f"\nTrue = {true_class}"
    if pred_class is not None:
        title += f" | Pred = {pred_class}"
    
    plt.title(title)
    plt.xlabel("2θ index")
    plt.ylabel("Normalized Importance")
    plt.tight_layout()
    plt.show()


def ig_entropy(importance):
    imp = np.abs(np.array(importance))  
    total = imp.sum()
    
    if total == 0:
        return 0.0
    
    imp = imp / total
    return -np.sum(imp * np.log(imp + 1e-12))

def top_regions(curve, top_k=6, half_width=10):
    c = np.abs(curve).copy()   
    n = len(c)
    regions = []

    for _ in range(top_k):
        center = int(np.argmax(c))
        
        if c[center] <= 0:
            break

        lo = max(0, center - half_width)
        hi = min(n - 1, center + half_width)

        regions.append((lo, hi, center, float(curve[center])))

        # suppress selected region
        c[lo:hi+1] = -1

    return regions


def compute_ig_for_sample(model, x_tensor, device, n_steps=25, target_class=None):
    model.eval()
    x_tensor = x_tensor.to(device).float()

    # predict class if not provided
    with torch.no_grad():
        _, _, logits, _ = model(x_tensor)
        pred_class = torch.argmax(logits, dim=1).item()

    if target_class is None:
        target_class = pred_class

    def forward_func(x):
        _, _, logits, _ = model(x)
        return logits

    ig = IntegratedGradients(forward_func)
    baseline = torch.zeros_like(x_tensor)

    attributions = ig.attribute(
        x_tensor,
        baselines=baseline,
        target=target_class,
        n_steps=n_steps
    )

    attrs_signed = attributions.squeeze().detach().cpu().numpy()
    importance = np.abs(attrs_signed)

    m = importance.max()
    if m > 0:
        importance = importance / m

    return pred_class, attrs_signed, importance


def build_class_mean_ig_from_arrays(ig_attributions, true_classes, num_classes, use_abs=True, normalize_each=False):
    """
    ig_attributions: (N, 2048) signed or unsigned
    true_classes:    (N,)
    use_abs:         True -> mean(|IG|) signatures (recommended)
    normalize_each:  True -> normalize each sample by its max before averaging (optional)
    """
    X = np.array(ig_attributions)

    if use_abs:
        X = np.abs(X)

    if normalize_each:
        denom = np.max(X, axis=1, keepdims=True)
        denom[denom == 0] = 1.0
        X = X / denom

    class_mean = np.zeros((num_classes, X.shape[1]), dtype=np.float32)

    for c in range(num_classes):
        mask = (true_classes == c)
        if np.any(mask):
            class_mean[c] = X[mask].mean(axis=0)

    return class_mean  # shape (num_classes, 2048)



def cosine_sim(a, b, eps=1e-12):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps))

def compare_with_class_signature(pred_class_idx, importance, class_names, class_mean_ig):
    """
    class_mean_ig: (num_classes, 2048)
    importance: (2048,)
    """
    class_signature = class_mean_ig[pred_class_idx]
    similarity = cosine_sim(importance, class_signature)
    return class_names[pred_class_idx], similarity, class_signature



def plot_comparison(
    importance,
    class_signature,
    class_name,
    similarity=None,
    theta_min=5,
    theta_max=90
):
    theta = np.linspace(theta_min, theta_max, len(importance))

    plt.figure(figsize=(14,4))
    plt.plot(theta, importance, label="New Sample IG", linewidth=2)
    plt.plot(theta, class_signature, label=f"Mean IG - {class_name}", alpha=0.8)

    title = f"IG Comparison for {class_name}"
    if similarity is not None:
        title += f" | Cosine Similarity = {similarity:.3f}"

    plt.title(title)
    plt.xlabel("2θ (degrees)")
    plt.ylabel("Normalized Importance")
    plt.legend()
    plt.tight_layout()
    plt.show()


def infer_sample_with_ig_nn(
    model,
    x_sample,
    class_names,
    train_igs,
    train_classes,
    device
):
    """
    Perform inference on a single XRD sample using:
      - Model logits
      - IG nearest-neighbor comparison

    Returns:
        dict with predictions and scores
    """

    model.eval()

    # Prepare input
    x_tensor = torch.tensor(x_sample[None, ...]).to(device)

    # ----------------------------
    # 1️ Model prediction
    # ----------------------------
    with torch.no_grad():
        _, _, logits, _ = model(x_tensor)
        probs = torch.softmax(logits, dim=1)

        confidence = probs.max().item()
        model_pred_idx = logits.argmax(dim=1).item()
        model_pred_name = class_names[model_pred_idx]

    # ----------------------------
    # 2️ Compute IG
    # ----------------------------
    _, importance = compute_ig_for_sample(
        model,
        x_tensor,
        device
    )

    importance = importance / (np.linalg.norm(importance) + 1e-8)

    # ----------------------------
    # 3️ IG Nearest Neighbor
    # ----------------------------
    sims = cosine_similarity(
        importance.reshape(1, -1),
        train_igs
    )[0]

    best_idx = np.argmax(sims)
    ig_pred_name = train_classes[best_idx]
    ig_similarity = sims[best_idx]

    return {
        "model_prediction": model_pred_name,
        "confidence": confidence,
        "ig_prediction": ig_pred_name,
        "ig_similarity": ig_similarity
    }

def build_class_mean_ig(ig_attributions, true_classes, num_classes):
    ig_abs = np.abs(ig_attributions)
    means = []

    for c in range(num_classes):
        mask = (true_classes == c)
        means.append(ig_abs[mask].mean(axis=0))

    return np.stack(means)
def plot_class_difference(class_mean_ig,
                          class_names,
                          class_A,
                          class_B,
                          theta_min=10,
                          theta_max=90):

    # Always interpret class_A/B as actual class values (NOT index)
    idx_A = list(class_names).index(class_A)
    idx_B = list(class_names).index(class_B)

    mean_A = class_mean_ig[idx_A]
    mean_B = class_mean_ig[idx_B]

    diff = mean_A - mean_B

    theta = np.linspace(theta_min, theta_max, len(diff))

    plt.figure(figsize=(14,4))
    plt.plot(theta, diff, color="black")

    plt.axhline(0, linestyle="--", linewidth=1)

    plt.fill_between(theta, 0, diff,
                     where=(diff > 0),
                     alpha=0.3,
                     label=f"{class_A} > {class_B}")

    plt.fill_between(theta, 0, diff,
                     where=(diff < 0),
                     alpha=0.3,
                     label=f"{class_B} > {class_A}")

    plt.xlabel("2θ (degrees)")
    plt.ylabel("Difference in Mean IG")
    plt.title(f"IG Discriminative Regions: {class_A} vs {class_B}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return diff

def attention_ratio(xrd, ig,two_theta, split_theta=45):
    idx_split = np.searchsorted(two_theta, split_theta)

    low = ig[:idx_split].mean()
    high = ig[idx_split:].mean()

    return low / high


def compute_ig_single(model, x, device, n_steps=32):

    baseline = torch.zeros_like(x).to(device)

    alphas = torch.linspace(0, 1, steps=n_steps).to(device)

    scaled_inputs = [
        baseline + alpha * (x - baseline)
        for alpha in alphas
    ]

    scaled_inputs = torch.cat(scaled_inputs, dim=0)
    scaled_inputs.requires_grad = True

    _, _, logits, _ = model(scaled_inputs)

    preds = logits.argmax(dim=1)
    selected = logits.gather(1, preds.unsqueeze(1)).squeeze()

    model.zero_grad()
    selected.sum().backward()

    grads = scaled_inputs.grad

    grads = grads.view(n_steps, -1)

    avg_grads = grads.mean(dim=0)

    ig = (x.squeeze(0) - baseline.squeeze(0)) * avg_grads

    ig = ig / (torch.norm(ig) + 1e-8)

    return ig



def compute_mean_ig_by_spacegroup(
    ig_vectors,            # (N, 2048)
    labels,                # (N,)
    sg_class_names         # [1,2,...,227] length 22
):
    sg_class_names = [int(sg) for sg in sg_class_names]
    sg_to_idx = {sg: i for i, sg in enumerate(sg_class_names)}
    num_classes = len(sg_class_names)

    ig_vectors = np.asarray(ig_vectors, dtype=np.float32)
    labels = np.asarray(labels)

    # collect sum/count by class index
    sums = {i: np.zeros(ig_vectors.shape[1], dtype=np.float32) for i in range(num_classes)}
    counts = {i: 0 for i in range(num_classes)}

    for ig, y in zip(ig_vectors, labels):
        yi = int(y)

        # y could be index 0..21 or SG number
        if 0 <= yi < num_classes:
            cls_idx = yi
        else:
            cls_idx = sg_to_idx.get(yi, None)
            if cls_idx is None:
                continue

        sums[cls_idx] += ig
        counts[cls_idx] += 1

    # build dict keyed by SG number
    mean_sg_igs = {}
    for i, sg in enumerate(sg_class_names):
        mean_sg_igs[sg] = (sums[i] / counts[i]) if counts[i] > 0 else None

    print("IG counts per SG:", {sg_class_names[i]: counts[i] for i in range(num_classes)})
    return mean_sg_igs

def run_single_inference(
    x_sample,
    model,
    class_names,
    train_igs_torch,
    train_classes_torch,
    device,
    top_k=3
):

    model.eval()

    # --------------------------
    # Prepare input
    # --------------------------
    x_tensor = torch.tensor(
        x_sample,
        dtype=torch.float32,
        device=device
    ).unsqueeze(0)

    # --------------------------
    # Model prediction
    # --------------------------
    with torch.no_grad():
        _, _, logits, _ = model(x_tensor)
        probs = torch.softmax(logits, dim=1)

        confidence, model_pred_idx = torch.max(probs, dim=1)
        model_pred_idx = model_pred_idx.item()
        confidence = confidence.item()

        model_pred_name = class_names[model_pred_idx]

        # Top-k model predictions
        topk_probs, topk_indices = torch.topk(probs, top_k)
        topk_model = [
            (class_names[idx.item()], prob.item())
            for idx, prob in zip(topk_indices[0], topk_probs[0])
        ]

    # --------------------------
    # Integrated Gradients
    # --------------------------
    ig_vector = compute_ig_single(
        model,
        x_tensor,
        device,
        n_steps=32
    )

    # --------------------------
    # Cosine similarity (GPU)
    # --------------------------
    sims = torch.matmul(train_igs_torch, ig_vector)

    # Best IG match
    best_idx = torch.argmax(sims)
    ig_pred_idx = train_classes_torch[best_idx].item()
    ig_pred_name = class_names[ig_pred_idx]
    similarity_score = sims[best_idx].item()

    # Top-k IG similarities
    topk_sims, topk_ig_indices = torch.topk(sims, top_k)
    topk_ig = [
        (class_names[train_classes_torch[idx].item()],
         sim.item())
        for idx, sim in zip(topk_ig_indices, topk_sims)
    ]

    return {
        "model_prediction": model_pred_name,
        "model_confidence": confidence,
        "model_topk": topk_model,
        "ig_prediction": ig_pred_name,
        "ig_similarity": similarity_score,
        "ig_topk": topk_ig
    }

def map_labels_to_sg_index(labels, sg_class_names):
    sg_class_names = [int(sg) for sg in sg_class_names]
    sg_to_idx = {sg: i for i, sg in enumerate(sg_class_names)}
    num_classes = len(sg_class_names)

    labels = np.asarray(labels)
    mapped = np.full(labels.shape, -1, dtype=np.int64)

    for i, y in enumerate(labels):
        yi = int(y)
        if 0 <= yi < num_classes:
            mapped[i] = yi
        else:
            mapped[i] = sg_to_idx.get(yi, -1)

    return mapped  # -1 means "unknown/not in sg_class_names"


# -------------------------------------------------
# 1) Compute mean importance per SG class
# -------------------------------------------------
def compute_mean_ig_by_sg(
    ig_train_igs,          # (N, 2048)
    ig_train_classes,      # (N,) indices or SG numbers
    sg_class_names,        # [1,2,...,227] length 22
    use_abs=True
):
    sg_class_names = [int(sg) for sg in sg_class_names]
    num_classes = len(sg_class_names)

    ig = np.asarray(ig_train_igs, dtype=np.float32)
    if use_abs:
        ig = np.abs(ig)

    y_idx = map_labels_to_sg_index(ig_train_classes, sg_class_names)

    mean_imp = []
    counts = []

    for c in range(num_classes):
        mask = (y_idx == c)
        n = int(mask.sum())
        counts.append(n)

        if n == 0:
            mean_imp.append(np.zeros(ig.shape[1], dtype=np.float32))
        else:
            mean_imp.append(ig[mask].mean(axis=0))

    mean_imp = np.stack(mean_imp, axis=0)  # (22, 2048)

    print("IG samples per SG:", {sg_class_names[i]: counts[i] for i in range(num_classes)})
    return mean_imp


# -------------------------------------------------
# 2) Run SG IG fingerprint analysis 
# -------------------------------------------------
def run_sg_ig_fingerprint_analysis(
    ig_train_igs,
    ig_train_classes,
    sg_class_names,
    top_regions_fn,            # your existing top_regions() function
    smooth_sigma=1,
    x_min=10,
    x_max=90
):
    sg_class_names = [int(sg) for sg in sg_class_names]
    num_classes = len(sg_class_names)

    # 1) Mean IG per class
    mean_imp = compute_mean_ig_by_sg(
        ig_train_igs=ig_train_igs,
        ig_train_classes=ig_train_classes,
        sg_class_names=sg_class_names,
        use_abs=True
    )  # (22, 2048)

    # 2) Build matrix + normalize per class
    M = mean_imp.copy()
    M_norm = M / (np.max(M, axis=1, keepdims=True) + 1e-8)

    # Optional smoothing
    if smooth_sigma is not None and smooth_sigma > 0:
        M_norm = gaussian_filter(M_norm, sigma=smooth_sigma)

    # 3) 2θ axis
    N = M.shape[1]
    two_theta = np.linspace(x_min, x_max, N)

    # 4) Print top regions per SG
    print("\nTop IG regions per SPACE GROUP:")

    for i, sg in enumerate(sg_class_names):
        regions = top_regions_fn(M_norm[i])

        print(f"\nSG: {sg}")
        for (lo, hi, center, score) in regions:
            print(
                f"  2θ [{two_theta[lo]:.2f}, {two_theta[hi]:.2f}] "
                f"(center={two_theta[center]:.2f})  score={score:.4f}"
            )

    # 5) IG heatmap
    plt.figure(figsize=(14, 7))
    im = plt.imshow(
        M_norm,
        aspect="auto",
        cmap="inferno",
        extent=[two_theta[0], two_theta[-1], 0, num_classes],
        origin="lower"
    )
    plt.colorbar(im, label="Normalized Mean IG Importance")
    plt.yticks(np.arange(num_classes) + 0.5,
            [str(sg) for sg in sg_class_names])
    plt.xlabel("2θ (degrees)")
    plt.ylabel("Space Group")

    for i in range(num_classes):
        plt.axhline(i, color="white", linewidth=0.3)

    plt.tight_layout()
    plt.savefig(IG_Heatmap_SG,
                format="pdf",
                bbox_inches="tight")
    plt.close()

    # 6) Distances
    cos_sim = cosine_similarity(M_norm)
    cos_dist = 1 - cos_sim

    corr_dist = pairwise_distances(M_norm, metric="correlation")

    # 7) Plot distance heatmaps
    #plt.figure(figsize=(8, 7))
    #im = plt.imshow(cos_dist, cmap="viridis")
    #plt.xticks(range(num_classes), [str(sg) for sg in sg_class_names], rotation=90)
    #plt.yticks(range(num_classes), [str(sg) for sg in sg_class_names])
    #plt.colorbar(im, label="Cosine Distance")
    #plt.tight_layout()
    #plt.show()

    #plt.figure(figsize=(8, 7))
    #im = plt.imshow(corr_dist, cmap="magma")
    #plt.xticks(range(num_classes), [str(sg) for sg in sg_class_names], rotation=90)
    #plt.yticks(range(num_classes), [str(sg) for sg in sg_class_names])
    #plt.colorbar(im, label="Correlation Distance")
    #plt.tight_layout()
    #plt.show()

    # 8) Hierarchical clustering
    Z = linkage(M_norm, method="ward")

    plt.figure(figsize=(14, 7))   # larger figure

    dendrogram(
        Z,
        labels=[str(sg) for sg in sg_class_names],
        leaf_font_size=14          # larger SG numbers
    )

    plt.xticks(rotation=90)        # prevent overlap
    plt.ylabel("Distance", fontsize=18)
    plt.xlabel("Space Group", fontsize=18)

    plt.tight_layout()
    plt.savefig(
        "IG_Cosine_SG_pdf",
        format="pdf",
        bbox_inches="tight"
    )
    plt.close()
    # 9) Mean inter-class distance
    upper_tri = cos_dist[np.triu_indices(num_classes, k=1)]
    print("\nMean cosine inter-class distance:", float(upper_tri.mean()))

    upper_tri_corr = corr_dist[np.triu_indices(num_classes, k=1)]
    print("Mean correlation inter-class distance:", float(upper_tri_corr.mean()))

    return {
        "M_norm": M_norm,
        "two_theta": two_theta,
        "cos_dist": cos_dist,
        "corr_dist": corr_dist
    }