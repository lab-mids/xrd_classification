from mp_api.client import MPRester
from pymatgen.core import Structure
import numpy as np
import os

def redownload_cifs_from_saved_ids(api_key, ids_npy="ids_gaussian_new.npy", out_dir="recovered_cifs"):
    os.makedirs(out_dir, exist_ok=True)

    ids = np.load(ids_npy, allow_pickle=True)

    with MPRester(api_key) as m:
        for entry in ids:
            try:
                # Extract MP ID (remove SG part)
                mp_id = entry.split("_")[0]

                doc = m.materials.summary.search(material_ids=[mp_id])[0]
                structure: Structure = doc.structure

                out_path = os.path.join(out_dir, f"{mp_id}.cif")
                structure.to(fmt="cif", filename=out_path)

                print(f"✔ Recovered {mp_id}")

            except Exception as e:
                print(f"❌ Failed {entry}: {e}")

from mp_api.client import MPRester
from pymatgen.core import Structure
import numpy as np
import os
import requests

def recover_all_cifs(api_key, ids_npy="ids_gaussian_new.npy", out_dir="recovered_cifs"):
    os.makedirs(out_dir, exist_ok=True)

    ids = np.load(ids_npy, allow_pickle=True)

    print(f"Total IDs: {len(ids)}")

    with MPRester(api_key) as m:
        for entry in ids:
            entry = str(entry)

            try:
                # ================= MP =================
                if entry.startswith("mp-"):
                    mp_id = entry
                    doc = m.materials.summary.search(material_ids=[mp_id])[0]
                    structure: Structure = doc.structure

                    out_path = os.path.join(out_dir, f"{mp_id}.cif")
                    structure.to(fmt="cif", filename=out_path)
                    print(f"✔ MP recovered: {mp_id}")

                # ================= COD =================
                elif entry.isdigit():
                    cod_id = entry
                    url = f"https://www.crystallography.net/cod/{cod_id}.cif"
                    out_path = os.path.join(out_dir, f"{cod_id}.cif")

                    r = requests.get(url, timeout=10)
                    if r.status_code == 200:
                        with open(out_path, "wb") as f:
                            f.write(r.content)
                        print(f"✔ COD recovered: {cod_id}")
                    else:
                        print(f"COD not found: {cod_id}")

            except Exception as e:
                print(f"Failed {entry}: {e}")

    print("\ All possible CIFs recovered.")