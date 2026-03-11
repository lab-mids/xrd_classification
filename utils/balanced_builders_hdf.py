import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
# balanced_builders.py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple

CS_NAMES = ["triclinic", "monoclinic", "orthorhombic",
            "tetragonal", "trigonal", "hexagonal", "cubic"]

import torch
from torch.utils.data import Dataset
import h5py
import numpy as np

class H5XRDDataset(Dataset):
    def __init__(self, h5_path, indices, labels):
        self.h5_path = h5_path
        self.indices = np.asarray(indices)
        self.labels = torch.from_numpy(labels.astype(np.int64))

        # Lazy open (per worker safe)
        self._h5 = None

    def _get_file(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        f = self._get_file()
        x = f["xrd"][self.indices[i]].astype(np.float32)

        # per-sample normalization (safe)
        m = x.max()
        if m > 0:
            x = x / m

        return torch.from_numpy(x), self.labels[i]

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

CS_NAMES = [
    "triclinic", "monoclinic", "orthorhombic",
    "tetragonal", "trigonal", "hexagonal", "cubic"
]

def cs_to_idx(name):
    return CS_NAMES.index(str(name).lower())


def build_balanced_cs_loaders_from_h5(
    h5_path,
    per_class_cs=7000,
    batch_size=64,
    val_split=0.1,
    test_split=0.1,
    seed=42,
    num_workers=4,
):
    with h5py.File(h5_path, "r") as f:
        cs = f["crystalsystems"][:].astype(str)   # ← CORRECT KEY
        n_points = f["xrd"].shape[1]

    y_all = np.array([cs_to_idx(s) for s in cs], dtype=np.int64)

    rng = np.random.default_rng(seed)
    keep_idx_parts = []

    for c in range(7):
        idx = np.where(y_all == c)[0]
        if len(idx) == 0:
            raise ValueError(f"No samples for {CS_NAMES[c]}")
        rng.shuffle(idx)
        keep_idx_parts.append(idx[:min(per_class_cs, len(idx))])

    keep_idx = np.concatenate(keep_idx_parts)
    y_keep = y_all[keep_idx]

    # -------- Stratified split --------
    pos_all = np.arange(len(keep_idx))

    pos_tr, pos_tmp, y_tr, y_tmp = train_test_split(
        pos_all,
        y_keep,
        test_size=(val_split + test_split),
        random_state=seed,
        stratify=y_keep,
    )

    test_frac = test_split / (val_split + test_split)

    pos_val, pos_te = train_test_split(
        pos_tmp,
        test_size=test_frac,
        random_state=seed,
        stratify=y_tmp,
    )

    # -------- Create datasets --------
    ds_tr = H5XRDDataset(h5_path, keep_idx[pos_tr], y_keep[pos_tr])
    ds_va = H5XRDDataset(h5_path, keep_idx[pos_val], y_keep[pos_val])
    ds_te = H5XRDDataset(h5_path, keep_idx[pos_te], y_keep[pos_te])

    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        ds_tr,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
    )

    val_loader = DataLoader(
        ds_va,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
    )

    test_loader = DataLoader(
        ds_te,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
    )

    counts = {CS_NAMES[c]: int((y_keep == c).sum()) for c in range(7)}

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "sizes": {
            "train": len(ds_tr),
            "val": len(ds_va),
            "test": len(ds_te),
        },
        "num_classes": 7,
        "class_names": CS_NAMES,
        "counts": counts,
        "input_len": n_points,
    }


import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
# balanced_builders.py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple


import torch
from torch.utils.data import Dataset
import h5py
import numpy as np

class H5XRDDataset(Dataset):
    def __init__(self, h5_path, indices, labels):
        self.h5_path = h5_path
        self.indices = np.asarray(indices)
        self.labels = torch.from_numpy(labels.astype(np.int64))

        # Lazy open (per worker safe)
        self._h5 = None

    def _get_file(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        f = self._get_file()
        x = f["xrd"][self.indices[i]].astype(np.float32)

        # per-sample normalization (safe)
        m = x.max()
        if m > 0:
            x = x / m

        return torch.from_numpy(x), self.labels[i]


def build_balanced_sg_loaders_from_h5(
    h5_path,
    min_count_sg=1000,
    per_class_cap_sg=1000,
    batch_size=64,
    val_split=0.1,
    test_split=0.1,
    seed=42,
    num_workers=4,
):
    with h5py.File(h5_path, "r") as f:
        sg_all = f["labels"][:].astype(np.int64)   
        n_points = f["xrd"].shape[1]

    # -------- Filter SGs by minimum count --------
    vals, counts = np.unique(sg_all, return_counts=True)
    keep_sgs = vals[counts >= min_count_sg]

    if len(keep_sgs) == 0:
        raise ValueError("No SGs satisfy min_count_sg.")

    # Map SG number → contiguous class index
    sg_to_idx = {int(sg): i for i, sg in enumerate(sorted(keep_sgs))}

    rng = np.random.default_rng(seed)
    keep_idx_parts = []

    for sg in keep_sgs:
        idx = np.where(sg_all == sg)[0]
        rng.shuffle(idx)
        keep_idx_parts.append(idx[:min(per_class_cap_sg, len(idx))])

    keep_idx = np.concatenate(keep_idx_parts)

    # Build balanced labels
    y_keep = np.array([sg_to_idx[int(sg_all[i])] for i in keep_idx], dtype=np.int64)

    # -------- Stratified split --------
    pos_all = np.arange(len(keep_idx))

    pos_tr, pos_tmp, y_tr, y_tmp = train_test_split(
        pos_all,
        y_keep,
        test_size=(val_split + test_split),
        random_state=seed,
        stratify=y_keep,
    )

    test_frac = test_split / (val_split + test_split)

    pos_val, pos_te = train_test_split(
        pos_tmp,
        test_size=test_frac,
        random_state=seed,
        stratify=y_tmp,
    )

    # -------- Create datasets --------
    ds_tr = H5XRDDataset(h5_path, keep_idx[pos_tr], y_keep[pos_tr])
    ds_va = H5XRDDataset(h5_path, keep_idx[pos_val], y_keep[pos_val])
    ds_te = H5XRDDataset(h5_path, keep_idx[pos_te], y_keep[pos_te])

    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        ds_tr,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
    )

    val_loader = DataLoader(
        ds_va,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
    )

    test_loader = DataLoader(
        ds_te,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
    )

    selected_counts = {
        int(sg): int(min(per_class_cap_sg, cnt))
        for sg, cnt in zip(vals, counts)
        if sg in keep_sgs
    }

    return {
        "train_loader": train_loader,
        "val_loader":   val_loader,
        "test_loader":  test_loader,
        "sizes": {
            "train": len(ds_tr),
            "val":   len(ds_va),
            "test":  len(ds_te),
        },
        "num_classes": len(keep_sgs),
        "class_names": sorted([int(sg) for sg in keep_sgs]),  # SG numbers
        "label_map": sg_to_idx,
        "counts": selected_counts,
        "input_len": n_points,
    }
