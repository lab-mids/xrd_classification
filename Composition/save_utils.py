import numpy as np
import torch
import pickle
import os


# ==============================================================
# SAVE / LOAD EMBEDDINGS
# ==============================================================
def save_embeddings(path, X_train_emb, X_test_emb, X_train, X_test):
    os.makedirs(path, exist_ok=True)

    np.save(f"{path}/X_train_emb.npy", X_train_emb)
    np.save(f"{path}/X_test_emb.npy", X_test_emb)

    with open(f"{path}/train_list.pkl", "wb") as f:
        pickle.dump(X_train, f)

    with open(f"{path}/test_list.pkl", "wb") as f:
        pickle.dump(X_test, f)

    #print(f"[Saved] Embeddings + train/test lists → {path}")


def load_embeddings(path):
    X_train_emb = np.load(f"{path}/X_train_emb.npy")
    X_test_emb = np.load(f"{path}/X_test_emb.npy")

    with open(f"{path}/train_list.pkl", "rb") as f:
        X_train = pickle.load(f)

    with open(f"{path}/test_list.pkl", "rb") as f:
        X_test = pickle.load(f)

    #print(f"[Loaded] Embeddings + lists from → {path}")

    return X_train_emb, X_test_emb, X_train, X_test


# ==============================================================
# SAVE / LOAD QWEN MODEL (using torch)
# ==============================================================
def save_qwen_model(model, tokenizer, path):
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    #print(f"[Saved] Qwen model + tokenizer → {path}")


def load_qwen_model(model_class, path, device):
    tokenizer = model_class.tokenizer_class.from_pretrained(path)
    model = model_class.model_class.from_pretrained(path).to(device)
    print(f"[Loaded] Qwen model from → {path}")
    return tokenizer, model
