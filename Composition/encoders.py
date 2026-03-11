import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import pickle
from Composition.chem_parse import *
from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from pymatgen.core import Composition

# ==========================================================
# COMPOSITION-LEVEL ENCODER (string embedding)
# ==========================================================
class QwenCompositionEncoder:
    def __init__(self,
                 model_name="Qwen/Qwen2.5-1.5B-Instruct",
                 device=None,
                 batch_size=16):

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

    def encode(self, compositions):
        all_emb = []

        for i in range(0, len(compositions), self.batch_size):
            batch = compositions[i:i + self.batch_size]

            #  chemically clean BEFORE tokenization
            batch = [clean_formula_chemically_no_space(str(c)) for c in batch]

            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            batch_emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            all_emb.append(batch_emb)

        return np.vstack(all_emb).astype("float32")



# ==========================================================
# ELEMENT-LEVEL ENCODER (chemically weighted)
# ==========================================================
class QwenElementEncoder:
    def __init__(self,
                 model_name="Qwen/Qwen2.5-1.5B-Instruct",
                 device=None,
                 cache_path="qwen_element_cache.pkl"):

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

        self.cache_path = cache_path
        self.cache = {}
        self.load_cache()

    def encode_element(self, element):
        if element in self.cache:
            return self.cache[element]

        inputs = self.tokenizer([element], return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)

        emb = out.last_hidden_state.mean(dim=1).cpu().numpy()[0]
        self.cache[element] = emb
        return emb

    def encode(self, compositions):
        out = []

        for comp in compositions:
            comp_dict = parse_formula(comp)
            total = sum(comp_dict.values())

            vec = None
            for el, amt in comp_dict.items():
                emb = self.encode_element(el)
                weighted = (amt / total) * emb
                vec = weighted if vec is None else vec + weighted

            out.append(vec)

        return np.vstack(out).astype("float32")

    def load_cache(self):
        try:
            with open(self.cache_path, "rb") as f:
                self.cache = pickle.load(f)
            print(f"[ElementEncoder] Loaded cache from {self.cache_path}")
        except FileNotFoundError:
            print(f"[ElementEncoder] No cache at {self.cache_path}")

    def save_cache(self):
        with open(self.cache_path, "wb") as f:
            pickle.dump(self.cache, f)
        #print(f"[ElementEncoder] Saved cache → {self.cache_path}")




# ==========================================================
# MAGPIE COMPOSITION ENCODER (classical baseline)
# ==========================================================
class MagpieCompositionEncoder:
    def __init__(self):
        self.featurizer = ElementProperty.from_preset("magpie", impute_nan=True)
        self.imputer = SimpleImputer(strategy="mean")
        self.scaler = StandardScaler()
        self._is_fitted = False

    def _featurize(self, compositions):
        features = []

        for c in compositions:
            try:
                c = normalize_formula(str(c))
                comp = Composition(c)
                features.append(self.featurizer.featurize(comp))
            except Exception:
                features.append([np.nan] * len(self.featurizer.feature_labels()))

        return np.array(features, dtype=float)

    def fit_transform(self, compositions):
        X = self._featurize(compositions)
        X = self.imputer.fit_transform(X)
        X = self.scaler.fit_transform(X)
        self._is_fitted = True
        return X.astype("float32")

    def transform(self, compositions):
        if not self._is_fitted:
            raise RuntimeError("Magpie encoder must be fit before transform")

        X = self._featurize(compositions)
        X = self.imputer.transform(X)
        X = self.scaler.transform(X)
        return X.astype("float32")
