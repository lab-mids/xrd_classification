import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from Composition.chem_parse import parse_formula


# ============================================================
# Dataset split utility
# ============================================================
def load_composition_dataset(compositions, seed=42):
    """
    Train/test split for composition-only datasets.
    """
    X_train, X_test = train_test_split(
        compositions,
        test_size=0.2,
        random_state=seed
    )
    return compositions, X_train, X_test


# ============================================================
# Cosine similarity search
# ============================================================
def cosine_search(query_vec, X_train_norm, X_train, top_k=5):
    """
    Perform cosine similarity search.
    """

    sims = cosine_similarity(query_vec, X_train_norm)[0]
    idx = np.argsort(sims)[::-1][:top_k]

    return pd.DataFrame({
        "index": idx,                      
        "composition": [X_train[i] for i in idx],
        "score": sims[idx],
    })



# ============================================================
# Chemical similarity (element overlap)
# ============================================================
def chemical_similarity(f1: str, f2: str) -> float:
    """
    Fraction of shared elements between two compositions.
    """
    comp1 = parse_formula(f1)
    comp2 = parse_formula(f2)

    e1 = set(comp1.keys())
    e2 = set(comp2.keys())

    if not e1:
        return 0.0

    return len(e1 & e2) / len(e1)


# ============================================================
# Accuracy@k utility
# ============================================================
def compute_accuracy(query_formula, retrieved):
    """
    Chemical accuracy@k based on element overlap ≥ 0.5.
    """
    scores = [chemical_similarity(query_formula, comp) for comp in retrieved]
    correct = sum(s >= 0.5 for s in scores)
    return correct / len(scores), scores
