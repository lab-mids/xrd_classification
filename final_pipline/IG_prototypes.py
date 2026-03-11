import h5py
import numpy as np
from tqdm import tqdm
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


def build_ig_class_prototypes(h5_path, num_classes=7, chunk=20000):

    with h5py.File(h5_path, "r") as f:
        A = f["attributions"]      # (N, L)
        y = f["true_classes"]      # (N,)

        N, L = A.shape

        sums = np.zeros((num_classes, L), dtype=np.float64)
        counts = np.zeros((num_classes,), dtype=np.int64)

        # --------------------------------------------------
        # Accumulate class sums
        # --------------------------------------------------
        for start in tqdm(range(0, N, chunk), desc="Building IG prototypes"):
            end = min(N, start + chunk)

            attrs = A[start:end].astype(np.float32)
            labs  = y[start:end].astype(np.int64)

            for c in range(num_classes):
                mask = (labs == c)
                if mask.any():
                    sums[c] += attrs[mask].sum(axis=0)
                    counts[c] += mask.sum()

        # --------------------------------------------------
        # Mean prototype per class
        # --------------------------------------------------
        protos = sums / np.clip(counts[:, None], 1, None)

        protos = protos - protos.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(protos, axis=1, keepdims=True)
        protos = protos / np.clip(norms, 1e-8, None)

    return protos.astype(np.float32), counts


def load_ig_class_prototypes_npy(npy_path, device=None):
    import torch
    protos = np.load(npy_path).astype(np.float32)
    t = torch.tensor(protos, dtype=torch.float32)
    if device is not None:
        t = t.to(device)
    return t
if __name__ == "__main__":
    
    h5_path = r"C:\Users\doaam\Downloads\PhD\XRD\Code_XRD\VAE\ig_database_streaming.h5"
    protos, counts = build_ig_class_prototypes(h5_path, num_classes=7)

    np.save(r"C:\Users\doaam\Downloads\PhD\XRD\Code_XRD\VAE\ig_class_prototypes.npy", protos)
    print("Saved prototypes. counts =", counts)