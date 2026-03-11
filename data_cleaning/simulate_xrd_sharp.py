import time
import os
import numpy as np
import h5py
from pymatgen.core import Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from multiprocessing import Pool, cpu_count
import multiprocessing as mp


param_dict = {
    "noise_std": 2e-3,
    "instrument_radius": 240.0,
}

kwargs = {
    "bkg_-1": (0.0, 0.03),
    "bkg_-2": (0.0, 0.05),
    "sample_height": (-0.1, 0.1),
}


def random_wavelength(rng):
    # Cu Kα, Mo Kα, synchrotron-like
    choices = [
        1.5406,     # Cu Kα
        1.5444,     # Cu Kα2
    ]
    return rng.choice(choices)

def pseudo_voigt(x, x0, I, sigma, eta):
    gaussian = np.exp(-((x - x0) ** 2) / (2 * sigma**2))
    lorentz  = 1 / (1 + ((x - x0) / sigma) ** 2)
    return I * (eta * lorentz + (1 - eta) * gaussian)


def add_ka_doublet(x, curve, rng):
    split = 0.1 + rng.uniform(-0.02, 0.02)  # deg
    return curve + 0.5 * shift_curve(x, curve, split)

def subtract_baseline(y, q=10):
    # subtract a low percentile as baseline (robust)
    b = np.percentile(y, q)
    y2 = y - b
    y2[y2 < 0] = 0
    return y2



def normalize(y):
    return y / np.max(y) if np.max(y) > 0 else y

def add_background(x, rng):
    t = (x - x.min()) / (x.max() - x.min())
    b0 = rng.uniform(0.0, 0.01)
    b1 = rng.uniform(-0.01, 0.01)
    b2 = rng.uniform(-0.005, 0.005)
    return np.clip(b0 + b1*t + b2*t**2, 0, None)

def shift_curve(x, y, shift):
    # shift > 0 moves peaks to higher 2θ
    return np.interp(x - shift, x, y, left=y[0], right=y[-1])


def apply_sample_displacement(two_theta, sample_height, R):
    theta = np.radians(two_theta / 2)
    delta = -2 * sample_height / R * np.cos(theta)
    return two_theta + np.degrees(delta)

def poly_background(x, rng, kwargs):
    a = rng.uniform(*kwargs["bkg_-1"])
    b = rng.uniform(*kwargs["bkg_-2"])
    t = (x - x.min()) / (x.max() - x.min())
    return a*t + b*t**2

def random_preferred_axis(structure, rng):
    axis = rng.normal(size=3)
    return axis / np.linalg.norm(axis)

def march_dollase_factor(hkl, structure, pref_axis, r):
    n = hkl_to_normal(structure, hkl)
    cos_alpha = np.clip(np.dot(n, pref_axis), -1, 1)
    sin_alpha = np.sqrt(1 - cos_alpha**2)

    denom = (r**2 * cos_alpha**2 + sin_alpha**2 / r)
    return denom**(-1.5)


def hkl_to_normal(structure, hkl):
    hkl = np.array(hkl, dtype=float).flatten()

    # --- Handle hexagonal 4-index notation (h k i l) ---
    if len(hkl) == 4:
        h, k, i, l = hkl
        # Convert to 3-index
        hkl = np.array([h, k, l])

    # --- Safety fallback ---
    if len(hkl) != 3:
        return np.array([0, 0, 1])

    L = structure.lattice.matrix
    G = np.linalg.inv(L).T
    n = G @ hkl

    norm = np.linalg.norm(n)
    return n / norm if norm > 0 else np.array([0, 0, 1])

def generate_realistic_peaks(two_theta, intensity, x, rng, structure, hkls):

    # --- NEW: lattice variation ---
    lattice_scale = 1 + rng.normal(0, 0.001)
    two_theta = two_theta * (1 / lattice_scale)

    curve = np.zeros_like(x)
    #  NEW: per-augmentation resolution factor
    resolution_scale = rng.uniform(0.7, 1)

    r = rng.uniform(0.95, 1.1)
    pref_axis = random_preferred_axis(structure, rng)

    global_zero_shift = rng.normal(0, 0.06)

    for p, I, hkl in zip(two_theta, intensity, hkls):
        theta = np.radians(p / 2)

        sigma_inst = 0.05 + 0.02*np.tan(theta)

        D = rng.uniform(200, 1000)
        sigma_size = (0.9 * 1.5406) / (D * np.cos(theta)) * 57.3

        strain = rng.uniform(0.0, 0.0005)
        sigma_strain = 4 * strain * np.tan(theta) * 57.3

        sigma = (sigma_inst + sigma_size + sigma_strain) * resolution_scale
        sigma = np.clip(sigma, 0.01, 0.3)


        eta = rng.uniform(0.1, 0.5)

        md_factor = march_dollase_factor(hkl, structure, pref_axis, r)
        I_mod = I * md_factor

        # 1) Grain statistics / packing variation (±8%)
        I_mod *= rng.normal(1.0, 0.08)

        # 2) Angle-dependent attenuation (detector + absorption)
        I_mod *= max(1 - 0.003 * p, 0.6)

        # 3) Random weak peak disappearance
        if I_mod < 0.02 and rng.random() < 0.4:
            continue

        local_shift = rng.normal(0, 0.04)
        p_shifted = p + local_shift + global_zero_shift

        split = 0.1 + rng.uniform(-0.02, 0.02)
        curve += pseudo_voigt(x, p_shifted, I_mod, sigma, eta)
        curve += pseudo_voigt(x, p_shifted + split, 0.5 * I_mod, sigma, eta)

    return curve


# Gaussian peak function
def create_gaussian_peak(x, peak_position, intensity, sigma):
    return intensity * np.exp(-0.5 * ((x - peak_position) / sigma) ** 2)

# Generate Gaussian broadened curve from (2θ, intensity)
def generate_gaussian_xrd_curve(two_theta_list, intensity_list, x, sigma):
    xrd_curve = np.zeros_like(x)
    for peak_position, intensity in zip(two_theta_list, intensity_list):
        xrd_curve += create_gaussian_peak(x, peak_position, intensity, sigma)
    return xrd_curve

# Normalize intensity
def normalize_intensity(y):
    return y / np.max(y) if np.max(y) > 0 else y
def augment_xrd_pattern(two_theta, intensity, hkls, structure, x_uniform, rng, param_dict, kwargs):
    """
    Realistic laboratory XRD simulation (physics + extra noise)
    """

    # --------------------------------------------------
    # 1) Preferred orientation (mild texture only)
    # --------------------------------------------------
    intensity = intensity * rng.uniform(0.95, 1.05)

    # --------------------------------------------------
    # 2) Sample displacement (geometry error)
    # --------------------------------------------------
    sample_height = rng.uniform(*kwargs["sample_height"])
    two_theta = apply_sample_displacement(
        two_theta, sample_height, param_dict["instrument_radius"]
    )

    # --------------------------------------------------
    # 3) Realistic peak profiles (pseudo-Voigt + texture)
    # --------------------------------------------------
    curve = generate_realistic_peaks(two_theta, intensity, x_uniform, rng, structure, hkls)


    # --- NEW: realistic instrument jitter per augmentation ---
    instrument_zero_error = rng.normal(0, 0.03)
    detector_jitter = rng.normal(0, 0.08)

    curve = shift_curve(x_uniform, curve, instrument_zero_error + detector_jitter)



    # --------------------------------------------------
    # 6) Mild physical background
    # --------------------------------------------------
    background = rng.uniform(0.002, 0.008) * np.exp(-x_uniform / 60)
    curve += background

    # --------------------------------------------------
    # 7) Counting statistics (REAL detector physics)
    # --------------------------------------------------
    scale = rng.uniform(8000, 30000)
    counts = curve * scale
    counts = rng.poisson(np.clip(counts, 0, None))
    curve = counts / scale

    # --------------------------------------------------
    # 8) EXTRA augmentation noise (your addition)
    # --------------------------------------------------
    extra_noise_std = rng.uniform(0.002, 0.01)
    curve += rng.normal(0, extra_noise_std, size=len(curve))

    if rng.random() < 0.3:
        n_spikes = rng.integers(3, 15)
        spike_positions = rng.integers(0, len(curve), n_spikes)
        curve[spike_positions] += rng.uniform(0.02, 0.1, n_spikes)

    curve[curve < 0] = 0

    # --------------------------------------------------
    # 9) Small electronic noise
    # --------------------------------------------------
    curve += rng.normal(0, 0.0015, size=len(curve))
    # --- intensity dependent detector noise ---
    detector_noise = rng.normal(0, 0.002 + 0.01 * curve, size=len(curve))
    curve += detector_noise

    # --------------------------------------------------
    # 10) Final baseline correction
    # --------------------------------------------------
    curve = subtract_baseline(curve, q=2)

    return normalize(curve)

def process_single_cif(args):
    fname, source, theta_range, n_patterns_per_cif = args
    print(f"Worker started: {fname}", flush=True)  # ADD THIS


    results = []

    try:
        path = os.path.join(source, fname)
        structure = Structure.from_file(path)

        sga = SpacegroupAnalyzer(structure)
        sg_num = sga.get_space_group_number()
        crystal_sys = sga.get_crystal_system()
        formula = structure.composition.reduced_formula

        rng = np.random.default_rng()

        for i in range(n_patterns_per_cif):
            wl = random_wavelength(rng)
            pattern = XRDCalculator(wavelength=wl).get_pattern(
                structure, two_theta_range=theta_range
            )

            two_theta = np.array(pattern.x)
            intensity = normalize(np.array(pattern.y))
            hkls = [tuple(p[0]["hkl"]) if p and "hkl" in p[0] else (0,0,1)
                    for p in pattern.hkls]

            xrd_curve = augment_xrd_pattern(
            two_theta, intensity, hkls, structure,
            X_UNIFORM, rng, param_dict, kwargs)


            results.append((xrd_curve.astype("float32"), sg_num,
                            f"{fname}_aug{i}", formula, crystal_sys))

    except Exception as e:
        print(f"Failed CIF: {fname} | {e}")

    return results
def simulate_xrd_to_hdf5(
    allowed_ids,
    base_dirs=["mp_cif"],
    out_file="xrd_dataset.h5",
    theta_range=(10, 90),
    num_points=2048,
    n_patterns_per_cif=50,
    max_cifs=None,
):

    import os
    from multiprocessing import Pool

    global X_UNIFORM
    X_UNIFORM = np.linspace(*theta_range, num_points)

    global param_dict, kwargs
    param_dict = {
        "noise_std": 2e-3,
        "instrument_radius": 240.0,
    }
    kwargs = {
        "bkg_-1": (0.0, 0.03),
        "bkg_-2": (0.0, 0.05),
        "sample_height": (-0.1, 0.1),
    }

    # ------------------------------
    # Build task list
    # ------------------------------
    tasks = []
    for source in base_dirs:
        for fname in os.listdir(source):
            if fname.endswith(".cif"):
                file_id = fname.replace(".cif", "")
                base_id = file_id.split("_")[0]
                if file_id in allowed_ids or base_id in allowed_ids:
                    tasks.append((fname, source, theta_range, n_patterns_per_cif))

    if max_cifs:
        tasks = tasks[:max_cifs]

    print(f"Total CIF tasks: {len(tasks)}")

    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
    print(f"Using {num_workers} parallel workers", flush=True)

    # ------------------------------
    # STREAMING HDF5 WRITE
    # ------------------------------
    with h5py.File(out_file, "w") as f:

        dset_xrd = f.create_dataset("xrd", (0, num_points), maxshape=(None, num_points),
                                    dtype="float32", chunks=(128, num_points), compression="gzip")
        dset_labels = f.create_dataset("labels", (0,), maxshape=(None,), dtype="int")
        dset_ids = f.create_dataset("ids", (0,), maxshape=(None,), dtype=h5py.string_dtype())
        dset_comp = f.create_dataset("compositions", (0,), maxshape=(None,), dtype=h5py.string_dtype())
        dset_crys = f.create_dataset("crystalsystems", (0,), maxshape=(None,), dtype=h5py.string_dtype())
        f.create_dataset("xaxis", data=X_UNIFORM)

        index = 0
        BATCH_SIZE = 128
        batch_xrd, batch_labels, batch_ids, batch_comp, batch_crys = [], [], [], [], []

        print("Starting pool...", flush=True)

        with Pool(processes=num_workers) as pool:
            for result_batch in pool.imap_unordered(process_single_cif, tasks):

                for (xrd_curve, sg_num, fid, formula, crystal_sys) in result_batch:
                    batch_xrd.append(xrd_curve)
                    batch_labels.append(sg_num)
                    batch_ids.append(fid)
                    batch_comp.append(formula)
                    batch_crys.append(crystal_sys)

                    if len(batch_xrd) >= BATCH_SIZE:
                        n = len(batch_xrd)

                        dset_xrd.resize((index + n, num_points))
                        dset_labels.resize((index + n,))
                        dset_ids.resize((index + n,))
                        dset_comp.resize((index + n,))
                        dset_crys.resize((index + n,))

                        dset_xrd[index:index+n] = batch_xrd
                        dset_labels[index:index+n] = batch_labels
                        dset_ids[index:index+n] = batch_ids
                        dset_comp[index:index+n] = batch_comp
                        dset_crys[index:index+n] = batch_crys

                        index += n
                        
                        batch_xrd.clear()
                        batch_labels.clear()
                        batch_ids.clear()
                        batch_comp.clear()
                        batch_crys.clear()

        # Write leftover
        if batch_xrd:
            n = len(batch_xrd)
            dset_xrd.resize((index + n, num_points))
            dset_labels.resize((index + n,))
            dset_ids.resize((index + n,))
            dset_comp.resize((index + n,))
            dset_crys.resize((index + n,))

            dset_xrd[index:index+n] = batch_xrd
            dset_labels[index:index+n] = batch_labels
            dset_ids[index:index+n] = batch_ids
            dset_comp[index:index+n] = batch_comp
            dset_crys[index:index+n] = batch_crys

            index += n

    print(f"\n Done. Saved {index:,} patterns to: {out_file}")


if __name__ == "__main__":
    allowed_ids = set(np.load(
        "balanced_ids_master.npy",
        allow_pickle=True
    ))

    simulate_xrd_to_hdf5(
        allowed_ids=allowed_ids,
        base_dirs=["recovered_cifs"],
        out_file="xrd_dataset_new.h5",
        theta_range=(10, 90),
        num_points=2048,
        n_patterns_per_cif=15,
    )
