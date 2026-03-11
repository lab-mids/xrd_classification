from final_pipline.hybrid_inference import *
import numpy as np
import matplotlib.pyplot as plt
import h5py
import numpy as np
import torch

def build_theta_axis(x, theta_min=10.0, theta_max=90.0):
    """
    Create 2θ axis matching XRD signal length.
    """
    L = len(x)
    return np.linspace(theta_min, theta_max, L)

def plot_inference_cs_sg(
    result_cs,
    result_sg,
    xaxis,
    cs_class_names,
    sg_class_names,
    save_path=None,
):
  

    x = np.asarray(result_cs["x_input"])
    x_hat = np.asarray(result_cs["x_reconstruction"])

    ig_cs = result_cs.get("ig_importance", None)
    ig_sg = result_sg.get("ig_importance", None)

    # -------------------------
    # normalize helper
    # -------------------------
    def normalize_ig(v):
        if v is None:
            return np.zeros_like(x)

        v = np.asarray(v)

        # interpolate if mismatch
        if len(v) != len(x):
            v = np.interp(
                np.linspace(0,1,len(x)),
                np.linspace(0,1,len(v)),
                v
            )

        v = np.abs(v)
        return v / (v.max() + 1e-8)

    ig_cs = normalize_ig(ig_cs)
    ig_sg = normalize_ig(ig_sg)

    # -------------------------
    # plotting style
    # -------------------------
    plt.rcParams.update({
        "font.size": 20,
        "axes.titlesize": 22,
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
        "figure.titlesize": 24,
        "figure.dpi": 150,
        "font.family": "sans-serif"
    })

    fig = plt.figure(figsize=(11,8))

    # =================================================
    # 1️ XRD + reconstruction
    # =================================================
    ax1 = plt.subplot(3,1,1)

    ax1.plot(xaxis, x, lw=1.8, label="Measured XRD")
    ax1.plot(xaxis, x_hat, "--", lw=1.6, label="Reconstruction")

    ax1.fill_between(xaxis, x, x_hat, alpha=0.25)

    ax1.set_title(
        f"CS: {cs_class_names[result_cs['final_pred']]} | "
        f"SG: {sg_class_names[result_sg['final_pred']]}"
    )

    ax1.legend(frameon=False)
    ax1.tick_params(labelbottom=False)
    ax1.set_ylabel("Intensity")

    # =================================================
    # helper: plot IG + top peaks
    # =================================================
    def plot_ig_panel(ax, ig, title, ylabel):

        ax.fill_between(xaxis, 0, ig, alpha=0.6)

        k = min(30, int(np.sum(ig > 0)))
        if k > 0:
            peak_idx = np.argsort(ig)[-k:]
            ax.scatter(
                np.asarray(xaxis)[peak_idx],
                ig[peak_idx],
                s=18,
                zorder=3,
                label="Top IG peaks"
            )

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # =================================================
    # 2️ CS IG
    # =================================================
    ax2 = plt.subplot(3,1,2, sharex=ax1)
    ax2.tick_params(axis="x", which="both", labelbottom=False)
    plot_ig_panel(
        ax2,
        ig_cs,
        "Integrated Gradients — Crystal System",
        "IG (CS)"
    )

    # =================================================
    # 3️ SG IG
    # =================================================
    ax3 = plt.subplot(3,1,3, sharex=ax1)
    plot_ig_panel(
        ax3,
        ig_sg,
        "Integrated Gradients — Space Group",
        "IG (SG)"
    )

    ax3.set_xlabel(r"2$\theta$ (degrees)")
    ax3.tick_params(axis="x", which="both", labelbottom=True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    plt.show()



def load_ig_db_to_gpu(h5_path, device):
    with h5py.File(h5_path, "r") as f:
        ig = f["attributions"][:]     # (N,L) float32
        y  = f["true_classes"][:]     # (N,) int

    ig = np.abs(ig).astype(np.float32)
    ig /= (np.linalg.norm(ig, axis=1, keepdims=True) + 1e-8)

    train_igs_torch = torch.tensor(ig, dtype=torch.float32, device=device)
    train_labels_torch = torch.tensor(y, dtype=torch.long, device=device)

    return train_igs_torch, train_labels_torch


def infer_cs_and_sg_once(
    # --- CS pipeline ---
    model_cls_cs,
    train_norm_cs,
    train_labels_cs,
    train_igs_torch_cs,
    train_ig_labels_torch_cs,
    cs_class_names,
    cs_class_to_idx,

    # --- SG pipeline ---
    model_cls_sg,
    train_norm_sg,
    train_labels_sg,
    train_igs_torch_sg,
    train_ig_labels_torch_sg,
    sg_class_names,
    sg_class_to_idx,

    # --- shared ---
    recon_model,
    x,
    device,

    # --- optional composition retrieval ---
    composition=None,
    search_engine=None,
    top_k_comp=5,

    # --- options ---
    compute_ig_mode="always",
    return_tops=True,
    top_n_model=5,
    top_n_retr=5,
    top_n_ig=5,

    # --- plotting ---
    do_plot=False,
    xaxis=None,                 # 1D
    plot_mode="cs",             # "cs" or "sg"
    save_path=None,
):
    """
    One call -> runs hybrid_inference twice:
      - CS head (7 classes)
      - SG head (22 classes)
    Also runs composition retrieval for CS + SG (optional).
    Returns one dict with `cs` and `sg` sub-results.
    """

    # -------------------------
    # 1) CS hybrid inference
    # -------------------------
    out_cs = hybrid_inference(
        classifier_model=model_cls_cs,
        recon_model=recon_model,
        x=x,
        train_norm=train_norm_cs,
        train_labels=train_labels_cs,
        device=device,
        train_igs_torch=train_igs_torch_cs,
        train_ig_labels_torch=train_ig_labels_torch_cs,
        composition=composition,
        search_engine=search_engine,
        class_to_idx=cs_class_to_idx,
        label_mode="cs",
        compute_ig_mode=compute_ig_mode,
        return_tops=return_tops,
        TOP_K=3,
    )

    # -------------------------
    # 2) SG hybrid inference
    # -------------------------
    out_sg = hybrid_inference(
        classifier_model=model_cls_sg,
        recon_model=recon_model,
        x=x,
        train_norm=train_norm_sg,
        train_labels=train_labels_sg,
        device=device,
        train_igs_torch=train_igs_torch_sg,
        train_ig_labels_torch=train_ig_labels_torch_sg,
        composition=composition,
        search_engine=search_engine,
        class_to_idx=sg_class_to_idx,
        label_mode="sg",
        compute_ig_mode=compute_ig_mode,
        return_tops=return_tops,
        TOP_K=3,
    )

    # -------------------------
    # 3) Composition retrieval (CS + SG) independent of heads
    # -------------------------
    comp = {
        "cs_pred": None,
        "sg_pred_idx": None,
        "sg_number": None,
        "tops": {}
    }

    if composition and search_engine is not None:
        # CS
        cs_res = query_by_composition_cs(
            composition_string=composition,
            search_engine=search_engine,
            class_to_idx=cs_class_to_idx,
            top_k=top_k_comp,
            return_top=return_tops,
            top_n=3
        )
        comp["cs_pred"] = cs_res.get("composition_pred", None)
        if return_tops:
            comp["tops"]["cs_top"] = cs_res.get("top_candidates", None)

        # SG
        sg_res = query_by_composition_sg(
            composition_string=composition,
            search_engine=search_engine,
            class_to_idx=sg_class_to_idx,
            top_k=top_k_comp,
            return_top=return_tops,
           top_n=3
        )
        comp["sg_pred_idx"] = sg_res.get("composition_pred", None)
        if comp["sg_pred_idx"] is not None:
            comp["sg_number"] = int(sg_class_names[int(comp["sg_pred_idx"])])

        if return_tops:
            comp["tops"]["sg_top_idx"] = sg_res.get("top_candidates", None)
            if comp["tops"]["sg_top_idx"] is not None:
                comp["tops"]["sg_top_decoded"] = [
                    {"sg": int(sg_class_names[int(t["label"])]), "prob": float(t["prob"])}
                    for t in comp["tops"]["sg_top_idx"]
                ]

    # -------------------------
    # 4) Optional plot
    # -------------------------
    if do_plot:

        if xaxis is None:
            xaxis = np.linspace(
                10.0,
                90.0,
                len(out_cs["x_input"])
            )

        plot_inference_cs_sg(
            result_cs=out_cs,
            result_sg=out_sg,
            xaxis=xaxis,
            cs_class_names=cs_class_names,
            sg_class_names=sg_class_names,
            save_path=save_path,
        )

    return {
        "cs": out_cs,
        "sg": out_sg,
        "composition": comp
    }
