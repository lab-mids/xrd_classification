import os
import json
import pandas as pd


this_dir = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(this_dir, "data")
SCRIPT_DIR = os.path.join(this_dir, "scripts")

# Folder data paths
DATA_RAW_PATH = os.path.join(DATA_DIR, "raw")
DATA_CLEAN_PATH = os.path.join(DATA_DIR, "clean")
DATA_RESULTS_PATH = os.path.join(DATA_DIR, "results")




# Clean data
x_path = os.path.join(DATA_CLEAN_PATH, "xrd_gaussian_new.npy")
sg_path = os.path.join(DATA_CLEAN_PATH, "labels_gaussian_new.npy")
cs_path = os.path.join(DATA_CLEAN_PATH, "crystalsystems_gaussian_new.npy")
comp_path = os.path.join(DATA_CLEAN_PATH, "compositions_gaussian_new.npy")
xrd_dataset = os.path.join(DATA_CLEAN_PATH, "xrd_dataset_sharp.h5")
external_test_dataset = os.path.join(DATA_CLEAN_PATH, "external_test_dataset.h5")
external_test_ruff_dataset = os.path.join(DATA_CLEAN_PATH, "external_test_ruff_dataset.h5")
ids_gaussian_new = os.path.join(DATA_CLEAN_PATH, "ids_gaussian_new.npy")

balanced_ids = os.path.join(DATA_CLEAN_PATH, "balanced_ids_master.npy")

# Results data
Inference_RESULT= os.path.join(DATA_RESULTS_PATH , "inference_result")
sample_indices= os.path.join(Inference_RESULT, "sample_indices.npy")
shap_train_angles= os.path.join(Inference_RESULT, "shap_train_angles.npy")
shap_train_labels= os.path.join(Inference_RESULT, "shap_train_labels.npy")
latent_dict= os.path.join(Inference_RESULT, "latent_dict.npz")
linked_vectors= os.path.join(Inference_RESULT, "linked_vectors.pkl")
SAVE_EMBEDDING= os.path.join(DATA_RESULTS_PATH , "saved_comp")
COMP_EMBEDDINGS= os.path.join(SAVE_EMBEDDING , "comp_embeddings")
ELEMENT_EMBEDDINGS= os.path.join(SAVE_EMBEDDING , "element_embeddings")
ELEMENT_EMBEDDINGS_FULL= os.path.join(SAVE_EMBEDDING , "element_embeddings_full.npy")
MAGPIE_EMBEDDINGS= os.path.join(SAVE_EMBEDDING , "magpie_embeddings")
result_compare= os.path.join(SAVE_EMBEDDING , "result")

#### CS Latent
CS_RESULTS = os.path.join(DATA_RESULTS_PATH, "CS")
CS_LATENT = os.path.join(CS_RESULTS, "cs_letent")
CS_Models= os.path.join(CS_RESULTS, "CS_models")
CS_Cls= os.path.join(CS_Models, "best_xrd_model_classifier_balanced_sharp_cs.pth")
CS_IG= os.path.join(CS_RESULTS, "IG")

IG_database_CS= os.path.join(CS_IG, "ig_database_streaming_new.h5")
IG_database_energy_CS= os.path.join(CS_IG, "ig_database_streaming_energy_CS.h5")
IG_Classes_Prototypes_CS= os.path.join(CS_IG, "ig_class_prototypes_CS.npy")
IG_Heatmap_CS= os.path.join(CS_IG, "ig_heatmap_cs.pdf")
IG_Cosine_CS= os.path.join(CS_IG, "cosine_dendrogram_cs.pdf")
CS_RECO= os.path.join(CS_Models, "final_xrd_model_reconstruction_balanced_sharp_cs.pth")

SG_RESULTS = os.path.join(DATA_RESULTS_PATH, "SG")
SG_LATENT = os.path.join(SG_RESULTS, "sg_letent")
SG_Models= os.path.join(SG_RESULTS, "SG_models")
SG_IG = os.path.join(SG_RESULTS, "IG")
IG_database_SG= os.path.join(SG_IG, "ig_database_streaming_SG.h5")
IG_database_energy_SG= os.path.join(SG_IG, "ig_database_streaming_SG_energy.h5")
SG_Cls= os.path.join(SG_Models, "best_xrd_model_sg_classifier_balanced_sharp.pth")

SG_rec= os.path.join(SG_Models, "final_xrd_model_sg_reconstruction_balanced_sharp.pth")
IG_Classes_Prototypes_SG= os.path.join(SG_IG, "ig_class_prototypes_SG.npy")
IG_Heatmap_SG= os.path.join(SG_IG, "ig_heatmap_sg.pdf")
IG_Cosine_SG= os.path.join(SG_IG, "cosine_dendrogram_sg.pdf")

Inference_Plot = os.path.join(DATA_RESULTS_PATH, "inference_plot")
Inference_Image = os.path.join(DATA_RESULTS_PATH, "Inference_Image.pdf")