import os
import numpy as np
import pandas as pd
from pymatgen.core import Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator


# -----------------------------
# Gaussian Peak Broadening
# -----------------------------
def gaussian(x, mu, amp, fwhm):
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return amp * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


# -----------------------------
# CIF → Continuous XRD Pattern
# -----------------------------
def cif_to_xrd_curve(cif_path, tth_grid, fwhm=0.3):
    structure = Structure.from_file(cif_path)
    calc = XRDCalculator(wavelength="CuKa")

    # Extract raw peaks
    pattern = calc.get_pattern(structure, two_theta_range=(tth_grid.min(), tth_grid.max()))
    peak_pos = np.array(pattern.x)
    peak_int = np.array(pattern.y)

    # Initialize continuous intensity curve
    intensity = np.zeros_like(tth_grid)

    # Add Gaussian-broadened peaks
    for mu, amp in zip(peak_pos, peak_int):
        intensity += gaussian(tth_grid, mu, amp, fwhm=fwhm)

    # Normalize
    intensity -= intensity.min()
    if intensity.max() > 0:
        intensity /= intensity.max()

    return peak_pos, peak_int, intensity


# -----------------------------
# CIF → CSV Converter Function
# -----------------------------
def convert_cifs_to_csv(folder, output_folder, fwhm=0.3, num_points=2048):
    """
    Convert all CIF files in 'folder' into XRD CSV files in 'output_folder'.

    CSV format:
        2theta, intensity
    """

    os.makedirs(output_folder, exist_ok=True)
    csv_files = []

    # Create a grid that matches your model input
    tth_grid = np.linspace(10, 90, num_points)

    for filename in os.listdir(folder):
        if not filename.lower().endswith(".cif"):
            continue

        cif_path = os.path.join(folder, filename)

        try:
            _, _, curve = cif_to_xrd_curve(cif_path, tth_grid, fwhm=fwhm)

            # Build output CSV name
            name = os.path.splitext(filename)[0]
            csv_path = os.path.join(output_folder, f"{name}_simulated_xrd.csv")

            # Save CSV
            df = pd.DataFrame({
                "2theta": tth_grid,
                "intensity": curve
            })
            df.to_csv(csv_path, index=False)

            print(f"✔ Saved CSV: {csv_path}")
            csv_files.append(csv_path)

        except Exception as e:
            print(f" Error processing {filename}: {e}")

    return csv_files