import os
import glob
import numpy as np
import nibabel as nib
from pathlib import Path
import concurrent.futures
from monai.transforms import (
    Compose,
    RandRotated,
    RandZoomd,
    Rand2DElasticd,
    EnsureChannelFirstd,
    SqueezeDimd,
    MapTransform
)

class AddChannel(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = np.expand_dims(d[key], axis=0)
        return d

def _process_file(file_path, out_path, aug_per_file, seed):
    np.random.seed(seed)
    # Define MONAI augmentation pipeline inside worker
    aug_transforms = Compose([
        AddChannel(keys=["label"]),
        RandRotated(
            keys=["label"], 
            range_x=0.3, # random rotation up to ~17 degrees
            prob=0.8, 
            mode=["nearest"], # MUST be nearest to preserve label integers
            padding_mode="zeros"
        ),
        RandZoomd(
            keys=["label"], 
            prob=0.5, 
            min_zoom=0.9, 
            max_zoom=1.1, 
            mode=["nearest"]
        ),
        Rand2DElasticd(
            keys=["label"],
            spacing=(20, 20),
            magnitude_range=(0.5, 1.5),  # conservative: subtle deformation only
            prob=0.5,
            mode=["nearest"],
            padding_mode="zeros"
        ),
        SqueezeDimd(keys=["label"], dim=0)
    ])

    file_path = Path(file_path)
    pose_folder = file_path.parent.parent.name # e.g. 'pose11'
    filename = file_path.name
    
    # Output directory for this pose
    pose_out_dir = out_path / pose_folder
    pose_out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load NIfTI
    img = nib.load(file_path)
    data = img.get_fdata()
    affine = img.affine
    
    # If the data is 3D but effectively 2D (e.g., shape (W, H, 1)), squeeze it
    data_2d = np.squeeze(data)
    
    if data_2d.ndim != 2:
        return 0
        
    # Save the original file to the augmented folder to keep everything together
    out_orig_path = pose_out_dir / f"orig_{filename}"
    nib.save(nib.Nifti1Image(data_2d.astype(np.int32), affine), out_orig_path)
    generated = 1
    
    # Generate augmented copies
    for i in range(aug_per_file):
        # Prepare data dict for MONAI
        data_dict = {"label": data_2d}
        
        # Apply transforms
        aug_dict = aug_transforms(data_dict)
        aug_data = aug_dict["label"]
        
        # Ensure it's still int32 for the labels
        aug_data = np.round(aug_data).astype(np.int32)
        
        # Save augmented NIfTI
        aug_filename = f"aug_{i:03d}_{filename}"
        out_aug_path = pose_out_dir / aug_filename
        nib.save(nib.Nifti1Image(aug_data, affine), out_aug_path)
        generated += 1
        
    return generated

def augment_segmentations(input_dir="segmentation", output_dir="segmentation_augmented", target_samples=500, seed=42):
    """
    Augments segmented NIfTI files to reach approximately `target_samples` in total.
    Applies random rotations and elastic deformations, preserving integer labels.
    """
    np.random.seed(seed)
    
    # Setup paths
    input_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Collect all segmented files from the subfolders
    search_pattern = str(input_path / "pose*" / "segmented" / "*.nii")
    all_files = glob.glob(search_pattern)
    
    if not all_files:
        print(f"No segmented files found in {search_pattern}")
        return
        
    print(f"Found {len(all_files)} original segmented files.")
    
    # Calculate how many augmentations to create per file to reach target
    # We always include the original, so we need (target_samples - len(all_files)) new ones.
    aug_per_file = max(1, int(np.ceil((target_samples - len(all_files)) / len(all_files))))
    print(f"Generating ~{aug_per_file} augmented copies per file...")
    
    total_generated = 0
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for i, file_path in enumerate(all_files):
            # Give each worker a slightly different seed
            worker_seed = seed + i
            futures.append(
                executor.submit(_process_file, file_path, out_path, aug_per_file, worker_seed)
            )
            
        for future in concurrent.futures.as_completed(futures):
            total_generated += future.result()
            if total_generated >= target_samples:
                # Cancel remaining futures if we hit target early
                for f in futures:
                    f.cancel()
                break
            
    print(f"Augmentation complete. Total files available for simulation: {total_generated}")

if __name__ == "__main__":
    augment_segmentations()
