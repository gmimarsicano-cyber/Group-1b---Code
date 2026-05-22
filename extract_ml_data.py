import os
import re
import glob
import numpy as np
from pathlib import Path


def _parse_pose_subject(filename):
    """Extract pose_id and subject_id from a filename of the form
    rf_volume_pose<ID>_..._sub<ID>_...npy. Returns (pose_id, sub_id) or None
    if the filename can't be parsed."""
    try:
        parts = filename.split("_")
        pose_part = next(p for p in parts if p.startswith("pose"))
        pose_id = int(pose_part.replace("pose", ""))
    except (StopIteration, ValueError):
        return None

    match = re.search(r"sub(\d+)", filename)
    sub_id = int(match.group(1)) if match else -1
    return pose_id, sub_id


def aggregate_simulation_results(input_dir="output/ultrasound_sim_batch",
                                 output_file="output/ultrasound_sim/wristband8_raw_sim_augmented.npz"):
    """
    Scans the input directory for individual RF volume .npy files,
    extracts the pose IDs from the filenames, and aggregates them into
    a single .npz file format expected by the ML pipeline.

    Streams files into a pre-allocated float32 buffer to keep peak RAM
    bounded (output_size + one file). The previous implementation held
    all files in a Python list before stacking, peaking near 2x the
    final array size and OOM-killing on the cluster.

    Expected filename format: rf_volume_pose<ID>_..._sub<ID>_...npy
    """
    input_path = Path(input_dir)
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    file_pattern = str(input_path / "rf_volume_*.npy")
    files = sorted(glob.glob(file_pattern))

    if not files:
        print(f"No simulation output files found in {file_pattern}")
        return

    print(f"Aggregating {len(files)} simulation results...")

    # Pass 1: parse metadata and read shapes via mmap (no data copy).
    keep = []  # list of (path, shape, pose_id, sub_id)
    for f in files:
        filename = os.path.basename(f)
        parsed = _parse_pose_subject(filename)
        if parsed is None:
            print(f"Could not parse pose/subject ID from {filename}, skipping...")
            continue
        pose_id, sub_id = parsed
        try:
            arr = np.load(f, mmap_mode="r")
        except Exception as e:
            print(f"Could not open {filename}, skipping... ({e})")
            continue
        keep.append((f, arr.shape, pose_id, sub_id))

    if not keep:
        print("No parseable files; nothing to aggregate.")
        return

    n_samples = len(keep)
    n_tx = keep[0][1][0]
    n_rx = keep[0][1][1]
    max_nt = max(shape[2] for _, shape, _, _ in keep)
    print(f"Pre-allocating output buffer: ({n_samples}, {n_tx}, {n_rx}, {max_nt}) float32 "
          f"= {n_samples * n_tx * n_rx * max_nt * 4 / 1e9:.2f} GB")

    scans_array = np.zeros((n_samples, n_tx, n_rx, max_nt), dtype=np.float32)
    pose_ids_array = np.empty(n_samples, dtype=np.int32)
    subject_ids_array = np.empty(n_samples, dtype=np.int32)

    # Pass 2: stream each file into its slot.
    for i, (f, shape, pose_id, sub_id) in enumerate(keep):
        arr = np.load(f).astype(np.float32, copy=False)
        scans_array[i, :, :, :arr.shape[2]] = arr
        pose_ids_array[i] = pose_id
        subject_ids_array[i] = sub_id
        del arr
        if (i + 1) % 25 == 0 or i + 1 == n_samples:
            print(f"  loaded {i + 1}/{n_samples}")

    print(f"Final aggregated scans shape: {scans_array.shape} (dtype={scans_array.dtype})")
    print(f"Final aggregated pose_ids shape: {pose_ids_array.shape}")
    print(f"Final aggregated subject_ids shape: {subject_ids_array.shape}")

    np.savez_compressed(
        out_path,
        scans=scans_array,
        pose_ids=pose_ids_array,
        subject_ids=subject_ids_array,
    )
    print(f"Saved aggregated simulation data to {out_path}")


if __name__ == "__main__":
    aggregate_simulation_results()
