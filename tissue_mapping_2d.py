import json
from pathlib import Path

import numpy as np
from PIL import Image

# If you don't have nibabel yet:
# pip install nibabel
import nibabel as nib
# ============================================================
# PART 1 — FUNCTIONS AND CONSTANTS (DON'T TOUCH)
# ============================================================

TISSUE_PROPERTIES = {
    0: {"name": "water-like medium", "sound_speed": 1500.0, "density": 1000.0, "alpha_coeff": 0.0, "alpha_power": 1.0},
    1: {"name": "bones",   "sound_speed": 3300.0, "density": 1970.0, "alpha_coeff": 12.0, "alpha_power": 1.5},
    2: {"name": "muscle",  "sound_speed": 1580.0, "density": 1070.0, "alpha_coeff": 1.1,  "alpha_power": 0.95},
    3: {"name": "tendon",  "sound_speed": 1750.0, "density": 1100.0, "alpha_coeff": 4.7,  "alpha_power": 0.79},
    4: {"name": "fat",     "sound_speed": 1475.0, "density": 937.0,  "alpha_coeff": 0.61, "alpha_power": 0.84},
    5: {"name": "joint",   "sound_speed": 1665.0, "density": 1100.0, "alpha_coeff": 5.0,  "alpha_power": 0.83},
}


def load_2d_labels(input_path):
    """
    Load a 2D labelmap with labels 0..5 only.

    Allowed inputs:
      - 2D NIfTI (.nii/.nii.gz)
      - 3D NIfTI with one singleton dimension (e.g., transversal slices)
      - 2D grayscale image (png/tif/jpg) where pixel values are 0..5

    Returns:
      labels: (H, W) int32 array with values in {0,1,2,3,4,5}
    """
    input_path = Path(input_path)
    suffix = "".join(input_path.suffixes).lower()

    # ---- 2D NIfTI ----
    if suffix.endswith(".nii") or suffix.endswith(".nii.gz"):
        img = nib.load(str(input_path))
        data = img.get_fdata(dtype=np.float32)

        # Squeeze out singleton dimensions (e.g., for transversal slices with shape (Z, 1, X))
        data = np.squeeze(data)
        
        if data.ndim != 2:
            raise ValueError(
                f"Expected a 2D NIfTI after squeezing, but got shape {data.shape}. "
                "This script only supports 2D inputs."
            )
        labels = np.rint(data).astype(np.int32)
    return labels

def build_property_maps(labels):
    """Build per-pixel physical property maps from the 2D labelmap."""
    h, w = labels.shape

    sound_speed = np.zeros((h, w), dtype=np.float32)
    density = np.zeros((h, w), dtype=np.float32)
    alpha_coeff = np.zeros((h, w), dtype=np.float32)
    alpha_power = np.zeros((h, w), dtype=np.float32)

    for label_id, props in TISSUE_PROPERTIES.items():
        mask = labels == label_id
        sound_speed[mask] = props["sound_speed"]
        density[mask] = props["density"]
        alpha_coeff[mask] = props["alpha_coeff"]
        alpha_power[mask] = props["alpha_power"]

    return sound_speed, density, alpha_coeff, alpha_power



def save_outputs(output_folder, labels, sound_speed, density, alpha_coeff, alpha_power):
    """Save outputs as .npy plus a preview image + JSON properties."""
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    np.save(output_folder / "label_map.npy", labels)
    np.save(output_folder / "sound_speed_map.npy", sound_speed)
    np.save(output_folder / "density_map.npy", density)
    np.save(output_folder / "alpha_coeff_map.npy", alpha_coeff)
    np.save(output_folder / "alpha_power_map.npy", alpha_power)

    # preview (grayscale)
    preview = (labels * 40).clip(0, 255).astype(np.uint8)
    Image.fromarray(preview).save(output_folder / "label_map_preview.png")

    with open(output_folder / "tissue_properties.json", "w", encoding="utf-8") as f:
        json.dump(TISSUE_PROPERTIES, f, indent=2)


def print_summary(labels):
    """Print pixel counts per label."""
    unique, counts = np.unique(labels, return_counts=True)
    print("Pixel counts by class:")
    for u, c in zip(unique, counts):
        print(f"  {int(u)} -> {TISSUE_PROPERTIES[int(u)]['name']}: {int(c)}")



def run_tissue_mapping_2d(input_path, output_folder="output_maps"):
    """Run the full workflow on a 2D labelmap."""
    labels = load_2d_labels(input_path)
    sound_speed, density, alpha_coeff, alpha_power = build_property_maps(labels)
    save_outputs(output_folder, labels, sound_speed, density, alpha_coeff, alpha_power)

    print("Done.")
    print(f"Input: {input_path}")
    print(f"Output folder: {output_folder}")
    print_summary(labels)
    


# # ============================================================
# # PART 2 — RUN (EDIT ONLY THIS)
# # ============================================================
# input_path = r"/data/leuven/387/vsc38717/Simulations/s_wrist_slice_transversal_POSE2sub1_01.nii.gz"
# output_folder = r"/data/leuven/387/vsc38717/Simulations/Output Mapping"

# run_tissue_mapping_2d(
#     input_path=input_path,
#     output_folder=output_folder
# )