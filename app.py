# app.py

import os
import sys
import h5py
import torch
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from numpy.linalg import norm
from config import *
# --------------------------------------------------
# Your project imports
# --------------------------------------------------

from final_pipline.hybrid_inference import *
from Composition.run_experiment import *
from Composition.query_composition import *
from final_pipline.full_inference import *
from final_pipline.output_table import *
from utils.balanced_builders_hdf import *
from models.autoencoder_classifier import *
# --------------------------------------------------
# Page config
# --------------------------------------------------

st.set_page_config(
    page_title="XRD Hybrid Inference",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 Hybrid XRD Inference Interface")
st.write("Crystal System and Space Group prediction from XRD pattern + composition.")


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def normalize_vectors(vectors):
    return vectors / np.clip(
        norm(vectors, axis=1, keepdims=True),
        1e-12,
        None
    )


def load_ig_db(h5_path):
    with h5py.File(h5_path, "r") as f:
        ig = f["attributions"][:]
        y = f["true_classes"][:]

    ig = np.abs(ig).astype(np.float32)
    ig /= np.linalg.norm(ig, axis=1, keepdims=True) + 1e-8

    train_igs_torch = torch.tensor(
        ig,
        dtype=torch.float32,
        device="cpu"
    )

    train_labels_torch = torch.tensor(
        y,
        dtype=torch.long,
        device="cpu"
    )

    return train_igs_torch, train_labels_torch


def plot_xrd(xrd):
    xaxis = np.linspace(10, 90, len(xrd))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(xaxis, xrd)

    ax.set_xlabel("2θ (degrees)")
    ax.set_ylabel("Intensity")
    ax.set_title("Input XRD Pattern")
    ax.grid(True, alpha=0.3)

    return fig

def read_xrd_csv(uploaded_file, target_length):
    df = pd.read_csv(uploaded_file)

    # find columns
    theta_col = None
    intensity_col = None

    for c in df.columns:
        c_lower = c.lower()

        if c_lower in ["two_theta", "2theta", "theta", "x"]:
            theta_col = c

        if c_lower in ["intensity", "i", "y", "xrd"]:
            intensity_col = c

    if theta_col is None:
        raise ValueError(
            "CSV must contain a two_theta column."
        )

    if intensity_col is None:
        raise ValueError(
            "CSV must contain an intensity column."
        )

    theta = df[theta_col].values.astype(np.float32)
    intensity = df[intensity_col].values.astype(np.float32)

    # keep only 10–90°
    mask = (theta >= 10.0) & (theta <= 90.0)

    theta = theta[mask]
    intensity = intensity[mask]

    if len(theta) < 10:
        raise ValueError(
            "No sufficient points found between 10° and 90°."
        )

    # resample to model length
    target_theta = np.linspace(10, 90, target_length)

    intensity_resampled = np.interp(
        target_theta,
        theta,
        intensity
    )

    return intensity_resampled, target_theta

def style_top_table(df, bg_color, header_color):
    return (
        df.style
        .set_table_styles([
            {
                "selector": "thead th",
                "props": [
                    ("background-color", header_color),
                    ("color", "white"),
                    ("font-weight", "bold"),
                    ("text-align", "center"),
                    ("font-size", "14px"),
                ]
            },
            {
                "selector": "tbody td",
                "props": [
                    ("background-color", bg_color),
                    ("color", "#222"),
                    ("font-size", "13px"),
                    ("text-align", "center"),
                    ("border", "1px solid #ddd"),
                ]
            }
        ])
        .format({"Score": "{:.5f}"})
    )
CS_NAMES = [
    "triclinic", "monoclinic", "orthorhombic",
    "tetragonal", "trigonal", "hexagonal", "cubic"
]

SG_CLASS_NAMES = [
    1, 2, 8, 11, 12, 14, 15, 19,
    38, 61, 62, 63, 71, 123, 139,
    166, 187, 194, 216, 221, 225, 227
]


def cs_name(label):
    try:
        return CS_NAMES[int(label)]
    except Exception:
        return str(label)


def sg_name(label):
    try:
        idx = int(label)
        if 0 <= idx < len(SG_CLASS_NAMES):
            return str(SG_CLASS_NAMES[idx])
        return str(label)
    except Exception:
        return str(label)


def top_table(items, label_type="cs"):
    rows = []

    for item in items:
        label = item.get("label")
        prob = item.get("prob", item.get("score_sum", item.get("score", None)))

        if label_type == "cs":
            label = cs_name(label)
        elif label_type == "sg":
            label = sg_name(label)

        rows.append({
            "Class": label,
            "Score": float(prob) if prob is not None else 0.0
        })

    return pd.DataFrame(rows)

def colored_header(title, color):
    st.markdown(
        f"""
        <div style="
            background:{color};
            color:white;
            padding:10px;
            border-radius:8px;
            text-align:center;
            font-weight:bold;
            margin-bottom:8px;">
            {title}
        </div>
        """,
        unsafe_allow_html=True
    )

def render_inference_result(res):
    st.subheader("Representative Inference Output")

    cs = res["cs"]
    sg = res["sg"]
    comp = res.get("composition", {})

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"""
            <div style="
                background:#E8F5E9;
                padding:20px;
                border-radius:12px;
                border-left:8px solid #4CAF50;">
                <h3>🟢 Crystal System</h3>
                <h2>{cs_name(cs["final_pred"])}</h2>
                <p><b>Model:</b> {cs_name(cs["model_pred"])}</p>
                <p><b>Retrieval:</b> {cs_name(cs["retrieval_pred"])}</p>
                <p><b>IG:</b> {cs_name(cs["ig_pred"])}</p>
                <p><b>Composition:</b> {cs_name(cs["composition_pred"])}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
   

    with col2:
        st.markdown(
            f"""
            <div style="
                background:#E3F2FD;
                padding:20px;
                border-radius:12px;
                border-left:8px solid #2196F3;">
                <h3>🔵 Space Group</h3>
                <h2>{sg_name(sg["final_pred"])}</h2>
                <p><b>Model:</b> {sg_name(sg["model_pred"])}</p>
                <p><b>Retrieval:</b> {sg_name(sg["retrieval_pred"])}</p>
                <p><b>IG:</b> {sg_name(sg["ig_pred"])}</p>
                <p><b>Composition:</b> {comp.get("sg_number", sg_name(sg["composition_pred"]))}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    

    st.markdown("### Additional Signals")

    col3, col4, col5 = st.columns(3)

    col3.metric(
        "Reconstruction Error",
        f"{cs['reconstruction_error']:.3e}"
    )

    col4.metric(
        "CS Entropy",
        f"{cs['entropy']:.5f}"
    )

    col5.metric(
        "SG Entropy",
        f"{sg['entropy']:.5f}"
    )

    st.markdown("### Top-k Candidates")

    tab1, tab2 = st.tabs(["Crystal System", "Space Group"])

    with tab1:
        c1, c2, c3 = st.columns(3)

        with c1:
            colored_header("Model", "#4CAF50")
            st.dataframe(
            style_top_table(
                top_table(cs["tops"]["model_top"], "cs"),
                bg_color="#E8F5E9",
                header_color="#4CAF50"
            ),
            use_container_width=True
        )

        with c2:
            colored_header("Retrieval", "#2196F3")
            st.dataframe(
                style_top_table(
                    top_table(cs["tops"]["retrieval_top"], "cs"),
                    bg_color="#E3F2FD",
                    header_color="#2196F3"
                ),
                use_container_width=True
            )
        with c3:
            colored_header("Composition", "#FF9800")
            st.dataframe(
            style_top_table(
                top_table(cs["tops"]["composition_top"], "cs"),
                bg_color="#FFF3E0",
                header_color="#FF9800"
            ),
            use_container_width=True
        )

        colored_header("Integrated Gradients", "#9C27B0")

        st.dataframe(
            style_top_table(
                top_table(cs["tops"]["ig_top"], "cs"),
                bg_color="#F3E5F5",
                header_color="#9C27B0"
            ),
            use_container_width=True
        )

    with tab2:

        s1, s2, s3 = st.columns(3)

        with s1:
            colored_header("Model", "#4CAF50")
            st.dataframe(
                style_top_table(
                    top_table(sg["tops"]["model_top"], "sg"),
                    bg_color="#E8F5E9",
                    header_color="#4CAF50"
                ),
                use_container_width=True
            )

        with s2:
            colored_header("Retrieval", "#2196F3")
            st.dataframe(
                style_top_table(
                    top_table(sg["tops"]["retrieval_top"], "sg"),
                    bg_color="#E3F2FD",
                    header_color="#2196F3"
                ),
                use_container_width=True
            )

        with s3:
            colored_header("Composition", "#FF9800")

            if "sg_top_decoded" in comp.get("tops", {}):

                df = pd.DataFrame(comp["tops"]["sg_top_decoded"])

                df = df.rename(columns={
                    "sg_number": "Class",
                    "prob": "Score"
                })

                st.dataframe(
                    style_top_table(
                        df,
                        bg_color="#FFF3E0",
                        header_color="#FF9800"
                    ),
                    use_container_width=True
                )

            else:

                st.dataframe(
                    style_top_table(
                        top_table(sg["tops"]["composition_top"], "sg"),
                        bg_color="#FFF3E0",
                        header_color="#FF9800"
                    ),
                    use_container_width=True
                )

        colored_header("Integrated Gradients", "#9C27B0")

        st.dataframe(
            style_top_table(
                top_table(sg["tops"]["ig_top"], "sg"),
                bg_color="#F3E5F5",
                header_color="#9C27B0"
            ),
            use_container_width=True
        )

    #st.markdown("### Composition Neighbors")

    #neighbors = cs["tops"].get("composition_neighbors", [])

    #if neighbors:
     #   df_neighbors = pd.DataFrame(neighbors)
    #    st.dataframe(df_neighbors, use_container_width=True)

    with st.expander("Show raw JSON"):
        st.json(res)
# --------------------------------------------------
# Cached loading
# --------------------------------------------------

@st.cache_resource
def load_all_assets():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------
    # Build CS loaders
    # --------------------------------------------------

    cs = build_balanced_cs_loaders_from_h5(
        h5_path=xrd_dataset,
        per_class_cs=105000,
        batch_size=256,
        val_split=0.1,
        test_split=0.1,
        seed=42,
        num_workers=4,
    )

    # --------------------------------------------------
    # Load metadata for composition search
    # --------------------------------------------------

    balanced_ids_arr = np.load(balanced_ids, allow_pickle=True)
    ids_arr = np.load(ids_gaussian_new, allow_pickle=True)
    compositions_arr = np.load(comp_path, allow_pickle=True)
    crystal_systems_arr = np.load(cs_path, allow_pickle=True)
    labels_arr = np.load(sg_path, allow_pickle=True)

    id_to_index = {id_: i for i, id_ in enumerate(ids_arr)}

    balanced_indices = np.array(
        [id_to_index[id_] for id_ in balanced_ids_arr],
        dtype=np.int64
    )

    compositions_balanced = compositions_arr[balanced_indices]
    ids_balanced = ids_arr[balanced_indices]
    labels_balanced = labels_arr[balanced_indices]
    balanced_cs = crystal_systems_arr[balanced_indices]

    _, unique_indices = np.unique(
        compositions_balanced,
        return_index=True
    )

    compositions_balanced = compositions_balanced[unique_indices]
    ids_balanced = ids_balanced[unique_indices]
    labels_balanced = labels_balanced[unique_indices]
    balanced_cs = balanced_cs[unique_indices]

    id_to_metadata = {
        real_id: {
            "composition": comp,
            "space_group": int(sg),
            "crystal_system": cs_name
        }
        for real_id, comp, sg, cs_name in zip(
            ids_balanced,
            compositions_balanced,
            labels_balanced,
            balanced_cs
        )
    }

    X_elem = np.load(ELEMENT_EMBEDDINGS_FULL)

    search_engine = ElementSearchEngine(
        ids=ids_balanced,
        id_to_metadata=id_to_metadata,
        embeddings=X_elem
    )

    # --------------------------------------------------
    # CS classifier
    # --------------------------------------------------

    model_cls_cs = DeepConvAutoencoderClassifier(
        input_length=cs["input_len"],
        latent_dim=64,
        cls_dim=128,
        num_classes=cs["num_classes"],
        use_projection_head=True
    ).to(device)

    state_dict = torch.load(
        CS_Cls,
        map_location=device,
        weights_only=True
    )

    model_cls_cs.load_state_dict(state_dict)
    model_cls_cs.eval()

    # --------------------------------------------------
    # CS reconstruction model
    # --------------------------------------------------

    model_rec_cs = DeepConvAutoencoderClassifier(
        input_length=cs["input_len"],
        latent_dim=64,
        cls_dim=128,
        num_classes=cs["num_classes"],
        use_projection_head=True
    ).to(device)

    state_dict = torch.load(
        CS_RECO,
        map_location=device,
        weights_only=True
    )

    model_rec_cs.load_state_dict(state_dict)
    model_rec_cs.eval()

    # --------------------------------------------------
    # SG loaders
    # --------------------------------------------------

    sg = build_balanced_sg_loaders_from_h5(
        h5_path=xrd_dataset,
        min_count_sg=20000,
        per_class_cap_sg=20000,
        batch_size=256,
        val_split=0.1,
        test_split=0.1,
        seed=42,
        num_workers=4,
    )

    # --------------------------------------------------
    # SG classifier
    # --------------------------------------------------

    model_cls_sg = DeepConvAutoencoderClassifier(
        input_length=sg["input_len"],
        latent_dim=64,
        cls_dim=128,
        num_classes=sg["num_classes"],
        use_projection_head=True
    ).to(device)

    checkpoint = torch.load(
        SG_Cls,
        map_location=device,
        weights_only=True
    )

    model_cls_sg.load_state_dict(checkpoint["model_state_dict"])
    model_cls_sg.eval()

    # --------------------------------------------------
    # SG reconstruction model
    # --------------------------------------------------

    model_rec_sg = DeepConvAutoencoderClassifier(
        input_length=sg["input_len"],
        latent_dim=64,
        cls_dim=128,
        num_classes=sg["num_classes"],
        use_projection_head=True
    ).to(device)

    checkpoint = torch.load(
        SG_Cls,
        map_location=device,
        weights_only=True
    )

    model_rec_sg.load_state_dict(checkpoint["model_state_dict"])
    model_rec_sg.eval()

    # --------------------------------------------------
    # Class mappings
    # --------------------------------------------------

    cs_class_names = [
        "triclinic",
        "monoclinic",
        "orthorhombic",
        "tetragonal",
        "trigonal",
        "hexagonal",
        "cubic"
    ]

    CS_CLASS_TO_IDX = {
        c: i for i, c in enumerate(cs_class_names)
    }

    sg_class_names = [
        1, 2, 8, 11, 12, 14, 15, 19,
        38, 61, 62, 63, 71, 123, 139,
        166, 187, 194, 216, 221, 225, 227
    ]

    SG_CLASS_TO_IDX = {
        int(sg_num): i for i, sg_num in enumerate(sg_class_names)
    }

    SG_CLASS_TO_IDX.update({
        str(int(sg_num)): i for i, sg_num in enumerate(sg_class_names)
    })

    # --------------------------------------------------
    # Load latent databases
    # --------------------------------------------------

    train_path_cs = os.path.join(
        CS_LATENT,
        "xrd_train_latents.npz"
    )

    train_data_cs = np.load(train_path_cs, allow_pickle=True)

    train_latents_cs = train_data_cs["latents"]
    train_labels_cs = train_data_cs["labels"]

    train_path_sg = os.path.join(
        SG_LATENT,
        "xrd_train_latents_sg.npz"
    )

    train_data_sg = np.load(train_path_sg, allow_pickle=True)

    train_latents_sg = train_data_sg["latents"]
    train_labels_sg = train_data_sg["labels"]

    train_norm_cs = normalize_vectors(train_latents_cs)
    train_norm_sg = normalize_vectors(train_latents_sg)

    # --------------------------------------------------
    # Load IG databases
    # --------------------------------------------------

    train_igs_torch_cs, train_ig_labels_torch_cs = load_ig_db(
        IG_database_CS
    )

    train_igs_torch_sg, train_ig_labels_torch_sg = load_ig_db(
        IG_database_SG
    )

    return {
        "device": device,

        "cs": cs,
        "sg": sg,

        "model_cls_cs": model_cls_cs,
        "model_rec_cs": model_rec_cs,

        "model_cls_sg": model_cls_sg,
        "model_rec_sg": model_rec_sg,

        "train_norm_cs": train_norm_cs,
        "train_labels_cs": train_labels_cs,
        "train_igs_torch_cs": train_igs_torch_cs,
        "train_ig_labels_torch_cs": train_ig_labels_torch_cs,

        "train_norm_sg": train_norm_sg,
        "train_labels_sg": train_labels_sg,
        "train_igs_torch_sg": train_igs_torch_sg,
        "train_ig_labels_torch_sg": train_ig_labels_torch_sg,

        "cs_class_names": cs_class_names,
        "CS_CLASS_TO_IDX": CS_CLASS_TO_IDX,

        "sg_class_names": sg_class_names,
        "SG_CLASS_TO_IDX": SG_CLASS_TO_IDX,

        "search_engine": search_engine
    }


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.header("Settings")

input_mode = st.sidebar.radio(
    "Input mode",
    [
        "Upload CSV",
        "Use external Ruff dataset sample"
    ]
)

do_plot = st.sidebar.checkbox(
    "Enable internal inference plots",
    value=False
)

sample_index = st.sidebar.number_input(
    "Sample index",
    min_value=0,
    value=260,
    step=1
)

composition_input = st.sidebar.text_input(
    "Composition",
    value="Ag-Au-Cu-Pd-Pt"
)


# --------------------------------------------------
# Load assets
# --------------------------------------------------

with st.spinner("Loading models and databases..."):
    assets = load_all_assets()

device = assets["device"]

st.success(f"Assets loaded. Using device: {device}")


# --------------------------------------------------
# Main input area
# --------------------------------------------------

x_sample_np = None
sample_composition = composition_input
true_cs = None
true_sg = None

if input_mode == "Upload CSV":

    uploaded_file = st.file_uploader(
        "Upload XRD CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            target_length = assets["cs"]["input_len"]

            x_sample_np, xaxis_uploaded = read_xrd_csv(
                uploaded_file,
                target_length
            )

            st.success(
                f"Loaded XRD pattern and resampled to {target_length} points."
            )

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(xaxis_uploaded, x_sample_np)

            ax.set_xlabel("2θ (degrees)")
            ax.set_ylabel("Intensity")
            ax.set_title("Input XRD Pattern")

            st.pyplot(fig)

        except Exception as e:
            st.error(f"Could not read uploaded file: {e}")

else:

    st.subheader("External Ruff Dataset")

    try:
        with h5py.File(external_test_ruff_dataset, "r") as f:

            x_test = f["xrd"][:]
            crystal_systems = f["crystalsystems"][:]
            space_groups = f["labels"][:]
            xaxis = f["xaxis"][:]
            compositions = f["compositions"][:]

        crystal_systems = [
            cs.decode() if isinstance(cs, bytes) else cs
            for cs in crystal_systems
        ]

        compositions = [
            c.decode() if isinstance(c, bytes) else c
            for c in compositions
        ]

        sg_class_names = assets["sg_class_names"]

        mask = np.isin(space_groups, sg_class_names)

        x_test = x_test[mask]
        crystal_systems = np.array(crystal_systems)[mask]
        compositions = np.array(compositions)[mask]
        space_groups = space_groups[mask]

        if sample_index >= len(x_test):
            st.error(
                f"Sample index is too large. Maximum index is {len(x_test) - 1}."
            )
        else:
            x_sample_np = x_test[sample_index]
            sample_composition = compositions[sample_index]
            true_cs = crystal_systems[sample_index]
            true_sg = space_groups[sample_index]

            col1, col2, col3 = st.columns(3)

            col1.metric("Sample index", sample_index)
            col2.metric("True Crystal System", str(true_cs))
            col3.metric("True Space Group", str(true_sg))

            st.write("Composition:", sample_composition)

            fig = plot_xrd(x_sample_np)
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Could not load external dataset: {e}")


# --------------------------------------------------
# Run inference
# --------------------------------------------------




run_button = st.button("🚀 Run Hybrid Inference")
if run_button:

    if x_sample_np is None:
        st.warning("Please upload or select an XRD sample first.")

    else:

        try:
            torch.cuda.empty_cache()

            x_sample = torch.tensor(
                x_sample_np,
                dtype=torch.float32,
                device=device
            ).unsqueeze(0)

            with st.spinner("Running hybrid inference..."):

                res = infer_cs_and_sg_once(
                # CS
                model_cls_cs=assets["model_cls_cs"],
                train_norm_cs=assets["train_norm_cs"],
                train_labels_cs=assets["train_labels_cs"],
                train_igs_torch_cs=assets["train_igs_torch_cs"],
                train_ig_labels_torch_cs=assets["train_ig_labels_torch_cs"],
                cs_class_names=assets["cs_class_names"],
                cs_class_to_idx=assets["CS_CLASS_TO_IDX"],

                # SG
                model_cls_sg=assets["model_cls_sg"],
                train_norm_sg=assets["train_norm_sg"],
                train_labels_sg=assets["train_labels_sg"],
                train_igs_torch_sg=assets["train_igs_torch_sg"],
                train_ig_labels_torch_sg=assets["train_ig_labels_torch_sg"],
                sg_class_names=assets["sg_class_names"],
                sg_class_to_idx=assets["SG_CLASS_TO_IDX"],

                recon_model=assets["model_rec_sg"],
                x=x_sample,
                device=device,

                composition=sample_composition,
                search_engine=assets["search_engine"],

                do_plot=do_plot
            )

            st.subheader("XRD Explainability")

            if do_plot and res.get("figure") is not None:
                st.pyplot(res["figure"])

            if isinstance(res, dict):
                render_inference_result(res)
            else:
                st.write(res)

        except Exception as e:
            st.error("Inference failed.")
            st.exception(e)

