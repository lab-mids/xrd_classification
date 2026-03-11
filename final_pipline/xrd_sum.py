from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import torch
import plotly.graph_objects as go
import plotly.express as px
from captum.attr import GradientShap


def plot_xrd_patterns_by_class(
    dataloader,
    class_names,
    max_patterns_per_class=200,
    alpha=0.05
):
    num_classes = len(class_names)
    collected = {i: [] for i in range(num_classes)}

    # Collect patterns
    for x, y in dataloader:
        x = x.cpu().numpy()
        y = y.cpu().numpy()

        for xi, yi in zip(x, y):
            if len(collected[yi]) < max_patterns_per_class:
                collected[yi].append(xi)

        if all(len(collected[i]) >= max_patterns_per_class for i in range(num_classes)):
            break

    # 2θ axis
    two_theta = np.linspace(10, 90, x.shape[1])

    # Color palette (color-blind friendly)
    colors = px.colors.qualitative.Safe

    for cls in range(num_classes):
        fig = go.Figure()
        class_color = colors[cls % len(colors)]

        for pattern in collected[cls]:
            fig.add_trace(
                go.Scatter(
                    x=two_theta,
                    y=pattern,
                    mode="lines",
                    line=dict(color=class_color, width=1),
                    opacity=alpha,
                    showlegend=False
                )
            )

        fig.update_layout(
            title=f"XRD patterns – {class_names[cls]}",
            xaxis_title="2θ (degrees)",
            yaxis_title="Intensity",
            xaxis=dict(range=[10, 90]),
            template="plotly_white",
            width=900,
            height=350
        )

        fig.show()


def plot_mean_xrd_patterns_by_class_separate(
    dataloader,
    class_names,
    max_patterns_per_class=200
):
    num_classes = len(class_names)
    collected = {i: [] for i in range(num_classes)}

    # Collect patterns
    for x, y in dataloader:
        x = x.cpu().numpy()
        y = y.cpu().numpy()

        for xi, yi in zip(x, y):
            if len(collected[yi]) < max_patterns_per_class:
                collected[yi].append(xi)

        if all(len(collected[i]) >= max_patterns_per_class for i in range(num_classes)):
            break

    # 2θ axis
    two_theta = np.linspace(10, 90, x.shape[1])

    colors = px.colors.qualitative.Safe  

    for cls in range(num_classes):
        patterns = np.array(collected[cls])
        mean_pattern = patterns.mean(axis=0)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=two_theta,
                y=mean_pattern,
                mode="lines",
                line=dict(
                    color=colors[cls % len(colors)],
                    width=3
                ),
                name=class_names[cls]
            )
        )

        fig.update_layout(
            title=f"Mean XRD pattern – {class_names[cls]}",
            xaxis_title="2θ (degrees)",
            yaxis_title="Mean intensity",
            xaxis=dict(range=[10, 90]),
            template="plotly_white",
            width=900,
            height=350
        )

        fig.show()

def compute_mean_xrd_by_class(dataloader, num_classes, max_patterns=7000):
    collected = {i: [] for i in range(num_classes)}

    for x, y in dataloader:
        x = x.cpu().numpy()
        y = y.cpu().numpy()

        for xi, yi in zip(x, y):
            if len(collected[yi]) < max_patterns:
                collected[yi].append(xi)

        if all(len(collected[i]) >= max_patterns for i in range(num_classes)):
            break

    mean_patterns = {}

    for cls in range(num_classes):
        patterns = np.array(collected[cls])
        mean_patterns[cls] = patterns.mean(axis=0)

    return mean_patterns

def compute_mean_xrd_by_spacegroup(
    dataloader,
    sg_class_names,          # e.g. [1,2,8,...,227]
    max_patterns=7000
):
    sg_to_idx = {int(sg): i for i, sg in enumerate(sg_class_names)}
    num_classes = len(sg_class_names)

    collected = {i: [] for i in range(num_classes)}

    for x, y in dataloader:
        x = x.cpu().numpy()
        y = y.cpu().numpy()

        for xi, yi in zip(x, y):
            yi_int = int(yi)

            # If yi is already an index, keep it
            if 0 <= yi_int < num_classes:
                cls_idx = yi_int
            else:
                # Otherwise treat it as SG number and map
                if yi_int not in sg_to_idx:
                    continue
                cls_idx = sg_to_idx[yi_int]

            if len(collected[cls_idx]) < max_patterns:
                collected[cls_idx].append(xi)

        if all(len(collected[i]) >= max_patterns for i in range(num_classes)):
            break

    mean_patterns_idx = {}
    for cls in range(num_classes):
        patterns = np.array(collected[cls])
        mean_patterns_idx[cls] = patterns.mean(axis=0) if len(patterns) else None

    # Return keyed by SG number (nicer)
    mean_patterns_sg = {
        int(sg_class_names[i]): mean_patterns_idx[i]
        for i in range(num_classes)
    }

    return mean_patterns_sg


def dominant_shap_peaks(peaks, tol=0.2, top_n=1000):
    """
    peaks: list or array of SHAP angles
    """
    peaks = np.sort(np.asarray(peaks))
    clusters = []
    counts = []

    for p in peaks:
        if not clusters:
            clusters.append(p)
            counts.append(1)
        elif abs(p - clusters[-1]) <= tol:
            counts[-1] += 1
            clusters[-1] = (clusters[-1] * (counts[-1] - 1) + p) / counts[-1]
        else:
            clusters.append(p)
            counts.append(1)

    clusters = np.array(clusters)
    counts = np.array(counts)

    idx = np.argsort(-counts)[:top_n]
    return clusters[idx], counts[idx]


def shap_peaks_from_saved(shap_angles, shap_labels, tol=0.2, top_n=100):
    shap_by_class = defaultdict(list)

    for angles, lbl in zip(shap_angles, shap_labels):
        shap_by_class[lbl].extend(angles)

    final = {}
    for cls, angles in shap_by_class.items():
        peaks, counts = dominant_shap_peaks(
            angles,
            tol=tol,
            top_n=top_n
        )
        final[cls] = peaks

    return final

# =========================================================
# Helper to extract logits from model
# =========================================================
def get_logits(inp, model, num_classes):
    out = model(inp)

    # Case 1 — direct tensor
    if isinstance(out, torch.Tensor) and out.shape[-1] == num_classes:
        return out

    # Case 2 — tuple/list output (e.g. (recon, logits))
    if isinstance(out, (tuple, list)):
        for o in out:
            if isinstance(o, torch.Tensor) and o.shape[-1] == num_classes:
                return o

    raise RuntimeError("No logits tensor found in model output!")


def compute_mean_xrd_by_spacegroup(
    dataloader,
    sg_class_names,          
    max_patterns=7000,
    verbose=True
):
    sg_class_names = [int(sg) for sg in sg_class_names]
    sg_to_idx = {sg: i for i, sg in enumerate(sg_class_names)}
    num_classes = len(sg_class_names)

    collected = {i: [] for i in range(num_classes)}
    used = {i: 0 for i in range(num_classes)}

    for x, y in dataloader:
        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()

        for xi, yi in zip(x, y):
            yi_int = int(yi)

            # yi could be an index (0..num_classes-1)
            if 0 <= yi_int < num_classes:
                cls_idx = yi_int
            else:
                # or yi could be the SG number (e.g., 225)
                cls_idx = sg_to_idx.get(yi_int, None)
                if cls_idx is None:
                    continue

            if used[cls_idx] < max_patterns:
                collected[cls_idx].append(xi)
                used[cls_idx] += 1

        if all(used[i] >= max_patterns for i in range(num_classes)):
            break

    mean_patterns_sg = {}
    for idx in range(num_classes):
        sg = sg_class_names[idx]
        if len(collected[idx]) == 0:
            mean_patterns_sg[sg] = None
        else:
            patterns = np.stack(collected[idx], axis=0)   # (N, 2048)
            mean_patterns_sg[sg] = patterns.mean(axis=0)  # (2048,)

    if verbose:
        counts = {sg_class_names[i]: used[i] for i in range(num_classes)}
        print("Collected patterns per SG:", counts)

    return mean_patterns_sg



def safe_norm(v):
    if v is None:
        return None
    vmax = float(np.max(np.abs(v)))
    if vmax < 1e-12:
        return v
    return v / vmax

def plot_mean_xrd_vs_mean_ig_by_sg(
    mean_sg_xrd,          # dict: SG -> (2048,)
    mean_sg_ig,           # dict: SG -> (2048,)
    sg_class_names,       # list of SG numbers, length 22
    x_min=10,
    x_max=90,
    title_prefix="SG",
    skip_missing=True
):
    sg_class_names = [int(sg) for sg in sg_class_names]

    # common 2θ axis (your setup)
    two_theta = np.linspace(x_min, x_max, 2048)

    for sg in sg_class_names:
        xrd = mean_sg_xrd.get(sg, None)
        ig  = mean_sg_ig.get(sg, None)

        if skip_missing and (xrd is None or ig is None):
            continue

        xrd_norm = safe_norm(xrd)
        ig_norm  = safe_norm(ig)

        plt.figure(figsize=(10, 4))
        plt.plot(two_theta, xrd_norm, linewidth=3, label="Mean XRD")
        plt.plot(two_theta, ig_norm, linestyle="--", linewidth=2, label="Mean Integrated Gradients")

        plt.title(f"{title_prefix} {sg} — Mean XRD vs IG")
        plt.xlabel("2θ (degrees)")
        plt.ylabel("Normalized intensity / importance")
        plt.legend()
        plt.tight_layout()
        plt.show()