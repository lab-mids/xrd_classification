import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from Composition.encoders import QwenElementEncoder
class ElementSearchEngine:

    def __init__(self, ids, id_to_metadata, embeddings):

        self.ids = ids
        self.id_to_metadata = id_to_metadata
        self.embeddings = embeddings
        self.encoder = QwenElementEncoder()

    def query(self, formula, top_k=5):

        query_vec = normalize(self.encoder.encode([formula]))
        sims = cosine_similarity(query_vec, self.embeddings)[0]
        idx = np.argsort(sims)[::-1][:top_k]

        rows = []

        for rank, i in enumerate(idx, start=1):
            real_id = self.ids[i]
            meta = self.id_to_metadata[real_id]

            rows.append({
                "rank": rank,
                "id": real_id,
                "composition": meta["composition"],
                "score": round(float(sims[i]), 4),
                "space_group": meta["space_group"],
                "crystal_system": meta["crystal_system"],
            })

        return pd.DataFrame(rows)
