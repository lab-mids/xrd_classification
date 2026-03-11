import torch
import numpy as np
import torch.nn.functional as F
from final_pipline.latent_similarty_funcs import *
from final_pipline.IG import *
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from collections import defaultdict
import torch

def query_by_composition_cs(
    composition_string,
    search_engine,
    class_to_idx,
    top_k=5,
    return_top=False,
    top_n=5,
):

    results = search_engine.query(composition_string, top_k=top_k)

    neighbors = []
    votes = []

    for _, row in results.iterrows():
        neighbors.append(row.to_dict())

        cs = row["crystal_system"]
        if cs in class_to_idx:
            votes.append(class_to_idx[cs])

    composition_pred = None
    top_candidates = None

    if len(votes) > 0:
        counts = np.bincount(votes)

        composition_pred = int(np.argmax(counts))

        if return_top:
            probs = counts / (counts.sum() + 1e-12)
            order = np.argsort(probs)[::-1][:top_n]

            top_candidates = [
            {"label": int(i), "prob": float(probs[i])}
            for i in order]

    return {
        "composition_pred": composition_pred,
        "neighbors": neighbors,
        "top_candidates": top_candidates,
    }

def query_by_composition_sg(
    composition_string,
    search_engine,
    class_to_idx,
    top_k=5,
    return_top=False,
    top_n=5,
):

    results = search_engine.query(composition_string, top_k=top_k)

    neighbors = []
    votes = []

    for _, row in results.iterrows():
        neighbors.append(row.to_dict())

        sg_val = row.get("space_group", None)
        if sg_val is None or np.isnan(sg_val):
            continue

        sg = int(sg_val)

        if sg in class_to_idx:
            votes.append(class_to_idx[sg])

    composition_pred = None
    top_candidates = None

    if len(votes) > 0:
        counts = np.bincount(votes)

        composition_pred = int(np.argmax(counts))

        if return_top:
            probs = counts / (counts.sum() + 1e-12)
            order = np.argsort(probs)[::-1][:top_n]

            top_candidates = [
            {"label": int(i), "prob": float(probs[i])}
            for i in order]

    return {
        "composition_pred": composition_pred,
        "neighbors": neighbors,
        "top_candidates": top_candidates,
    }

def adaptive_cosine_retrieval_classwise_topN(
    saved_latents,
    saved_labels,
    sample_latent,
    k_range=(50, 100, 200, 300, 500, 800, 1000),
    top_n=3,
    min_avg_cosine=0.5,
    temperature=0.2,          
    clamp_neg=True
):


    train_norm = saved_latents      # assumed normalized
    sample_norm = sample_latent     # assumed normalized

    cosine_scores = train_norm @ sample_norm
    sorted_idx = np.argsort(cosine_scores)[::-1]

    best_ranked = None  # keep best k result (by max prob)

    for k in k_range:
        idx_k = sorted_idx[:k]
        if len(idx_k) == 0:
            continue

        sims = cosine_scores[idx_k].astype(float)
        avg_cos = float(np.mean(sims))
        if avg_cos < min_avg_cosine:
            break

        class_sum = defaultdict(float)
        for idx, sim in zip(idx_k, sims):
            if clamp_neg:
                sim = max(sim, 0.0)
            class_sum[saved_labels[idx]] += float(sim)

        labels = list(class_sum.keys())
        scores = np.array([class_sum[l] for l in labels], dtype=float)

        # softmax with temperature
        z = scores / max(temperature, 1e-6)
        z = z - np.max(z)
        p = np.exp(z) / (np.sum(np.exp(z)) + 1e-12)

        order = np.argsort(p)[::-1]
        top = []
        for j in order[:top_n]:
            top.append({
                "label": labels[j],
                "k": k,
                "prob": float(p[j]),
                "score_sum": float(scores[j]),
                "avg_cosine_similarity": round(avg_cos, 4),
            })

        current = {
            "predicted_label": top[0]["label"],
            "top_candidates": top,
            "best_prob": top[0]["prob"],
            "k": k,
        }

        # choose the k that gives the most decisive retrieval
        if best_ranked is None or current["best_prob"] > best_ranked["best_prob"]:
            best_ranked = current

    if best_ranked is None:
        return {"predicted_label": None, "top_candidates": []}

    return {
        "predicted_label": best_ranked["predicted_label"],
        "top_candidates": best_ranked["top_candidates"],
    }
def ig_full_db_retrieval_topN(
    train_igs_torch,        
    train_labels_torch,     
    ig_vec_torch,           
    k=200,
    top_n=3,
    temperature=0.2,
    clamp_neg=True,
    chunk_size=10000,       
):

    # move IG vector to CPU (very small)
    ig_vec = ig_vec_torch.detach().cpu().float()
    ig_vec = F.normalize(ig_vec, p=2, dim=0)

    N = train_igs_torch.shape[0]

    all_sims = []

    # ---- chunked cosine ----
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)

        chunk = train_igs_torch[start:end]  # stays on CPU
        sims = torch.matmul(chunk, ig_vec)  # CPU matmul
        all_sims.append(sims)

    sims = torch.cat(all_sims, dim=0)

    # ---- top-k ----
    k = min(k, sims.numel())
    vals, idxs = torch.topk(sims, k=k)

    class_sum = defaultdict(float)
    for sim_val, idx in zip(vals, idxs):
        s = float(sim_val.item())
        if clamp_neg:
            s = max(s, 0.0)
        lbl = int(train_labels_torch[idx].item())
        class_sum[lbl] += s

    labels = list(class_sum.keys())
    scores = np.array([class_sum[l] for l in labels], dtype=np.float32)

    if len(scores) == 0:
        return {"predicted_label": None, "top_candidates": []}

    z = scores / max(temperature, 1e-6)
    z = z - np.max(z)
    p = np.exp(z) / (np.sum(np.exp(z)) + 1e-12)

    order = np.argsort(p)[::-1]
    top = []
    for j in order[:top_n]:
        top.append({
            "label": int(labels[j]),
            "prob": float(p[j]),
            "score_sum": float(scores[j]),
            "k": int(k),
        })

    return {
        "predicted_label": top[0]["label"],
        "top_candidates": top
    }

def compute_ig_vector_for_db_match(model, x, device, n_steps=32, use_abs=True):
    baseline = torch.zeros_like(x).to(device)
    alphas = torch.linspace(0, 1, steps=n_steps, device=device)

    # SAFE broadcasting
    alphas = alphas.view(-1, *([1] * x.dim()))
    scaled_inputs = baseline + alphas * (x - baseline)
    scaled_inputs = scaled_inputs.view(-1, *x.shape[1:])
    scaled_inputs.requires_grad_(True)

    _, _, logits, _ = model(scaled_inputs)

    preds = logits.argmax(dim=1)
    selected = logits.gather(1, preds.unsqueeze(1)).squeeze()

    model.zero_grad(set_to_none=True)
    selected.sum().backward()

    grads = scaled_inputs.grad
    grads = grads.view(n_steps, -1)
    avg_grads = grads.mean(dim=0)

    ig = (x.view(-1) - baseline.view(-1)) * avg_grads

    if use_abs:
        ig = ig.abs()

    ig = ig / (ig.norm(p=2) + 1e-8)
    return ig

def hybrid_inference(
    classifier_model,
    recon_model,
    x,
    train_norm,
    train_labels,
    device,
    train_igs_torch=None,
    train_ig_labels_torch=None,
    composition=None,
    search_engine=None,
    class_to_idx=None,
    recon_threshold=0.02,
    w_model=0.55,
    w_retr=0.30,
    w_ig=0.10,
    w_comp=0.05,
    agree_bonus=0.40,
    ig_sigmoid_scale=4.0,
    compute_ig_mode="auto",
    return_tops=False,
    TOP_K=3,
    label_mode="sg",
):

    classifier_model.eval()
    recon_model.eval()

    # =========================
    # INPUT
    # =========================
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32)

    if x.ndim == 1:
        x = x.unsqueeze(0)

    x = x.to(device)

    x_input_np = x.detach().cpu().numpy()[0].astype(np.float32)

    # =========================
    # MODEL
    # =========================
    with torch.no_grad():

        _, z_cls, logits, _ = classifier_model(x, return_latent=True)

        probs = torch.softmax(logits, dim=1)

        model_conf, model_pred = torch.max(probs, dim=1)

    model_pred = int(model_pred.item())
    model_conf = float(model_conf.item())

    model_top = None
    if return_tops:
        vals, idxs = torch.topk(probs[0], k=min(TOP_K, probs.shape[1]))
        model_top = [
            {"label": int(i), "prob": float(v)}
            for v, i in zip(vals, idxs)
        ]

    # =========================
    # LATENT RETRIEVAL
    # =========================
    sample_latent = z_cls.detach().cpu().numpy()[0]
    sample_latent = sample_latent / (np.linalg.norm(sample_latent) + 1e-12)

    adaptive_results = adaptive_cosine_retrieval_classwise_topN(
        saved_latents=train_norm,
        saved_labels=train_labels,
        sample_latent=sample_latent,
        top_n=TOP_K
    )

    retrieval_pred = adaptive_results.get("predicted_label")

    if retrieval_pred is None:
        retrieval_pred = model_pred

    retrieval_top = adaptive_results.get("top_candidates", []) or []

    retrieval_conf = (
        float(retrieval_top[0]["prob"]) if len(retrieval_top) > 0 else 0.0
    )

    retrieval_conf = max(0.0, min(1.0, retrieval_conf))

    # =========================
    # COMPOSITION
    # =========================
    composition_pred = None
    composition_neighbors = None
    composition_top = None

    if (
        composition is not None
        and search_engine is not None
        and class_to_idx is not None
    ):

        if label_mode == "sg":

            comp_res = query_by_composition_sg(
                composition_string=composition,
                search_engine=search_engine,
                class_to_idx=class_to_idx,
                return_top=True,
                top_n=TOP_K
            )

        else:

            comp_res = query_by_composition_cs(
                composition_string=composition,
                search_engine=search_engine,
                class_to_idx=class_to_idx,
                return_top=True,
                top_n=TOP_K
            )

        composition_pred = comp_res.get("composition_pred")

        if return_tops:
            composition_neighbors = comp_res.get("neighbors")
            composition_top = comp_res.get("top_candidates")

    # =========================
    # RECONSTRUCTION
    # =========================
    with torch.no_grad():

        x_hat = recon_model(x)

        if isinstance(x_hat, tuple):
            x_hat = x_hat[0]

        recon_error = float(torch.mean((x_hat - x) ** 2).item())

    x_hat_np = x_hat.detach().cpu().numpy()[0].astype(np.float32)

    # =========================
    # IG
    # =========================

    entropy = float((-(probs * torch.log(probs + 1e-8))).sum(dim=1).item())

    need_ig = False

    if compute_ig_mode == "always":
        need_ig = True
    elif compute_ig_mode == "auto":
        need_ig = (entropy > 0.30) or (retrieval_pred != model_pred)

    ig_pred = None
    ig_similarity = None
    ig_attr_vector = None
    ig_top = None

    if need_ig and train_igs_torch is not None:

        ig_vec = compute_ig_vector_for_db_match(
            model=classifier_model,
            x=x,
            device=device,
            n_steps=32,
            use_abs=True,
        )

        ig_attr_vector = ig_vec.detach().cpu().numpy().astype(np.float32)

        ig_db_res = ig_full_db_retrieval_topN(
            train_igs_torch=train_igs_torch,
            train_labels_torch=train_ig_labels_torch,
            ig_vec_torch=ig_vec,
            k=200,
            top_n=TOP_K
        )

        ig_pred = ig_db_res["predicted_label"]
        ig_top = ig_db_res["top_candidates"]

        ig_similarity = float(ig_top[0]["prob"]) if ig_top else None

    # =========================
    # RESULT DICT
    # =========================
    result = {

        "final_pred": None,

        "model_pred": model_pred,
        "model_confidence": model_conf,

        "retrieval_pred": int(retrieval_pred),
        "retrieval_confidence": retrieval_conf,

        "ig_pred": ig_pred,
        "ig_similarity": ig_similarity,

        "composition_pred": composition_pred,

        "reconstruction_error": recon_error,
        "entropy": entropy,

        "used_override": False,

        "x_input": x_input_np,
        "x_reconstruction": x_hat_np,
        "ig_importance": ig_attr_vector,

        "tops": {
            "model_top": model_top if return_tops else None,
            "retrieval_top": retrieval_top if return_tops else None,
            "ig_top": ig_top if return_tops else None,
            "composition_top": composition_top if return_tops else None,
            "composition_neighbors": composition_neighbors if return_tops else None,
        },
    }

    # =========================
    # AGREEMENT OVERRIDE
    # =========================
    if (
        composition_pred is not None
        and composition_pred == retrieval_pred
        and retrieval_pred != model_pred
    ):
        result["final_pred"] = retrieval_pred
        result["used_override"] = True
        return result

    # =========================
    # FUSION
    # =========================

    num_classes = probs.shape[1]
    score = torch.zeros(num_classes, device=device)

    if label_mode == "sg":

        if retrieval_pred != model_pred:

            if retrieval_conf >= 0.35:

                result["final_pred"] = int(retrieval_pred)
                result["used_override"] = True
                return result

    model_weight = float(np.exp(-recon_error * 10.0))

    # model distribution contribution
    score += w_model * model_weight * (model_conf ** 2) * probs[0]

    # retrieval contribution (sharpened)
    if len(retrieval_top) > 0:

        for cand in retrieval_top:

            lbl = int(cand["label"])
            p = float(cand["prob"])

            p = max(0.0, min(1.0, p))

            score[lbl] += w_retr * (p ** 1.5)

    else:
        score[int(retrieval_pred)] += w_retr * retrieval_conf

    # composition vote
    if composition_pred is not None:
        score[int(composition_pred)] += w_comp

    # IG vote
    if ig_pred is not None and ig_similarity is not None:

        ig_w = torch.sigmoid(
            torch.tensor(ig_sigmoid_scale * ig_similarity, device=device)
        ).item()

        score[int(ig_pred)] += w_ig * ig_w

    final_pred = int(torch.argmax(score))

    # anomaly rescue
    if recon_error > recon_threshold and ig_pred is not None and model_conf < 0.7:
        final_pred = int(ig_pred)

    result["final_pred"] = final_pred

    return result

# ============================================================
# RUN MATERIAL INFERENCE (wrapper)
# ============================================================
def run_material_inference(
    classifier_model=None,
    recon_model=None,
    x=None,
    composition=None,
    train_norm=None,
    train_labels=None,
    search_engine=None,
    device=None,
    class_to_idx=None,
    label_mode="sg",   
    compute_ig_mode="always",
    return_tops=True,
    TOP_K = 3,
    train_igs_torch=None,
    train_ig_labels_torch=None,
):
    if x is None and composition is None:
        raise ValueError("Provide x or composition.")

    return hybrid_inference(
        classifier_model=classifier_model,
        recon_model=recon_model,
        x=x,
        train_norm=train_norm,
        train_labels=train_labels,
        device=device,
        train_igs_torch=train_igs_torch,
        train_ig_labels_torch=train_ig_labels_torch,
        composition=composition,
        search_engine=search_engine,
        class_to_idx=class_to_idx,
        label_mode=label_mode,
        compute_ig_mode=compute_ig_mode,
        return_tops=return_tops,
        TOP_K=TOP_K,
    )

# ============================================================
# PLOT (robust + publication-ready)
# ============================================================
def plot_inference_paper(
    result,
    xaxis,
    class_names,
    save_path=None
):
    import numpy as np
    import matplotlib.pyplot as plt

    x = result.get("x_input", None)
    x_hat = result.get("x_reconstruction", None)

    if x is None or x_hat is None:
        raise ValueError("Result does not contain plotting diagnostics (x_input/x_reconstruction).")

    x = np.asarray(x)
    x_hat = np.asarray(x_hat)

    ig = result.get("ig_importance", None)

    # IG safe handling
    if ig is None:
        ig = np.zeros_like(x)
    else:
        ig = np.asarray(ig)
        if len(ig) != len(x):
            ig = np.interp(
                np.linspace(0, 1, len(x)),
                np.linspace(0, 1, len(ig)),
                ig
            )

    ig = np.abs(ig)
    ig = ig / (ig.max() + 1e-8)

    # residual (for visualization)
    residual = np.abs(x - x_hat)
    residual = residual / (residual.max() + 1e-8)
    residual = np.clip(residual, 0, 1)

    # Style
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 11,
        "figure.dpi": 150,
        "font.family": "sans-serif"
    })

    fig = plt.figure(figsize=(10, 6))

    # Top panel: XRD + reconstruction + residual shading
    ax1 = plt.subplot(2, 1, 1)

    ax1.plot(xaxis, x, linewidth=1.8, label="Measured XRD")
    ax1.plot(xaxis, x_hat, linestyle="--", linewidth=1.6, label="Reconstruction")

    ax1.fill_between(xaxis, x, x_hat, alpha=0.25, label="Reconstruction error")

    ax1.set_ylabel("Intensity (a.u.)")
    ax1.legend(frameon=False)
    ax1.set_xticks([])

    title = (
        f"{class_names[result['final_pred']]} | "
        f"Conf={result['model_confidence']:.2f} | "
        f"ReconErr={result['reconstruction_error']:.4f}"
    )
    #ax1.set_title(title)

    # Bottom panel: IG
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)

    ax2.fill_between(xaxis, 0, ig, alpha=0.6, label="IG Attribution")

    # stable peak selection
    k = min(40, int(np.sum(ig > 0)))
    if k > 0:
        peak_idx = np.argsort(ig)[-k:]
        ax2.scatter(np.asarray(xaxis)[peak_idx], ig[peak_idx], s=18, zorder=3, label="Top IG peaks")

    ax2.set_xlabel(r"2$\theta$ (degrees)")
    ax2.set_ylabel("IG Importance")

    ax2.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(frameon=False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.savefig(save_path.replace(".png", ".pdf"), bbox_inches="tight")

    plt.show()