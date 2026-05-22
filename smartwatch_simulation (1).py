import os
import shutil
import tempfile
import uuid
from pathlib import Path

import numpy as np

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

from scipy.ndimage import rotate as nd_rotate
from scipy.ndimage import zoom as nd_zoom

from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.ksensor import kSensor
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2D
from kwave.options.simulation_options import SimulationOptions
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from src.kwave.pulse import gaussian_pulse, apply_apodization

from src.kwave.tissue_mapping_2d import load_2d_labels, build_property_maps

# ============================================================
# CONSTANTS (From standalone smartwatch_simulation.py SECTION 0)
# ============================================================
TONE_FREQ = 5e6
TONE_CYCLES = 3
SOURCE_PA = 2e5

# Smartwatch probe geometry
N_RECEIVERS = 4
RECEIVER_WIDTH = 13
RECEIVER_HEIGHT = 1
RECEIVER_PITCH_PX = 15

TRANSMITTER_WIDTH = 25
TRANSMITTER_HEIGHT = 1
TX_RX_GAP_PX = 2

# Clustered array parameters — kept for grid-spacing computation only
N_CLUSTERS = 8
ELEMENTS_PER_CLUSTER = 4
ELEMENT_WIDTH = 5
ELEMENT_HEIGHT = 1
INTRA_CLUSTER_PITCH_PX = 7
CLUSTER_EDGE_OVERHANG_PX = 5
MIN_INTER_CLUSTER_EDGE_GAP_PX = 4

FIXED_PROBE_OFFSET_PX = 30
PPW = 4
C_REF_GRID = 1475.0

BG_C = 1500.0
BG_RHO = 1000.0
BG_ALPHA = 0.002
ALPHA_POWER = 0.8

CANONICAL_AXIS = "horizontal"
ENABLE_BONE_SIDE_CORRECTION = True
BONE_LABEL_ID = 1
ENABLE_BONE_DEPTH_ALIGNMENT = False
REFERENCE_BONE_CENTER_ROW = 1850  # dormant; kept for reference

TARGET_WRIST_WIDTH_MM = 55
WRIST_SIZE_METRIC = "bounding_box_width"

REFERENCE_CANVAS_NX = 64   # lower bound only; ensure_reference_canvas_size computes the real size
REFERENCE_CANVAS_NY = 64   # lower bound only
REFERENCE_TOP_MARGIN_PX = 80  # 80px × 0.15mm = 12mm water coupling above skin
REFERENCE_CENTER_COL = None
REFERENCE_CENTER_ROW = None

PML_INSIDE = True
PML_SIZE = 20
PML_ALPHA = 2.0
MIN_PML_TRANSDUCER_WAVELENGTHS = 2.0

CFL = 0.3
SAVE_TO_DISK = True

APPLY_EARLY_TIME_CUTOFF = True
EARLY_TIME_CUTOFF_MARGIN_PX = 5
DIRECT_WAVE_CUTOFF_MARGIN_PX = 5
EARLY_TIME_CUTOFF_EXTRA_SAMPLES = 20


# ============================================================
# HELPER FUNCTIONS (verbatim from clustered_simulation.py)
# ============================================================

def force_2d_labels(raw: np.ndarray) -> np.ndarray:
    arr = np.squeeze(raw).astype(np.int32)
    if arr.ndim != 2:
        raise ValueError(f"Label array is {arr.ndim}-D after squeezing.")
    return arr


def register_labels_to_canonical_axis(labels: np.ndarray, canonical: str = "vertical") -> tuple:
    pts = np.argwhere(labels > 0).astype(np.float64)
    if pts.shape[0] < 2:
        return labels, 0.0
    pts -= pts.mean(axis=0)
    cov = np.cov(pts.T)
    eigenvalues, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, -1]
    angle_from_horizontal = np.degrees(np.arctan2(principal[0], principal[1]))
    target_angle = 90.0 if canonical == "vertical" else 0.0
    angle_deg = target_angle - angle_from_horizontal
    if angle_deg > 90:
        angle_deg -= 180.0
    elif angle_deg < -90:
        angle_deg += 180.0
    labels_reg = nd_rotate(
        labels,
        angle=angle_deg,
        axes=(0, 1),
        reshape=True,
        order=0,
        cval=0,
        prefilter=False,
    )
    return labels_reg.astype(np.int32), angle_deg


def enforce_bones_in_lower_half(labels: np.ndarray, bone_label_id: int = 1) -> tuple:
    anatomy_pts = np.argwhere(labels > 0)
    bone_pts = np.argwhere(labels == bone_label_id)
    if anatomy_pts.size == 0 or bone_pts.size == 0:
        return labels, {}
    anatomy_centroid_row = float(anatomy_pts[:, 0].mean())
    bone_centroid_row = float(bone_pts[:, 0].mean())
    if bone_centroid_row < anatomy_centroid_row:
        labels_corrected = np.rot90(labels, 2).astype(np.int32)
        rotation_applied_deg = 180
    else:
        labels_corrected = labels
        rotation_applied_deg = 0
    return labels_corrected, {"rotation_applied_deg": rotation_applied_deg}


def compute_dx_from_ppw(c_min: float, f_max: float, ppw: int):
    lam_min = c_min / f_max
    dx = lam_min / ppw
    return dx, dx, lam_min


def compute_required_wrist_pixels_for_array_geometry(
    n_clusters,
    elements_per_cluster,
    element_width,
    intra_cluster_pitch_px,
    cluster_edge_overhang_px,
    min_inter_cluster_edge_gap_px,
):
    cluster_width_px = (elements_per_cluster - 1) * intra_cluster_pitch_px + element_width
    if n_clusters == 1:
        required_wrist_width_px = max(1, cluster_width_px - 2 * cluster_edge_overhang_px)
        center_spacing_px = 0.0
    else:
        center_spacing_px = cluster_width_px + min_inter_cluster_edge_gap_px
        required_wrist_width_px = int(
            np.ceil(
                cluster_width_px
                - 2 * cluster_edge_overhang_px
                + (n_clusters - 1) * center_spacing_px
            )
        )
    return {
        "cluster_width_px": int(cluster_width_px),
        "required_wrist_width_px": int(required_wrist_width_px),
        "min_inter_cluster_edge_gap_px": int(min_inter_cluster_edge_gap_px),
        "center_spacing_px": float(center_spacing_px),
    }


def choose_grid_spacing_for_geometry(
    c_min, f_max, base_ppw, target_wrist_width_mm, required_wrist_width_px
):
    dx_base, dy_base, lam_min = compute_dx_from_ppw(c_min, f_max, base_ppw)
    dx_geometry = (target_wrist_width_mm * 1e-3) / required_wrist_width_px
    dx_min = 0.10e-3  # 0.15 mm
    dx_chosen = max(dx_min, min(dx_base, dx_geometry))
    return {
        "dx": dx_chosen,
        "dy": dx_chosen,
        "lam_min": lam_min,
        "dx_base": dx_base,
        "dx_geometry": dx_geometry,
        "effective_ppw": lam_min / dx_chosen,
        "required_wrist_width_px": required_wrist_width_px,
    }


def pad_to_fft_friendly(arr: np.ndarray, pad_value: float = 0.0) -> np.ndarray:
    def next_smooth(n: int) -> int:
        candidate = n
        while True:
            tmp = candidate
            for p in (2, 3, 5):
                while tmp % p == 0:
                    tmp //= p
            if tmp == 1:
                return candidate
            candidate += 1

    nx_target, ny_target = next_smooth(arr.shape[0]), next_smooth(arr.shape[1])
    pad_x, pad_y = nx_target - arr.shape[0], ny_target - arr.shape[1]
    px0, px1 = pad_x // 2, pad_x - pad_x // 2
    py0, py1 = pad_y // 2, pad_y - pad_y // 2
    return np.pad(
        arr,
        pad_width=((px0, px1), (py0, py1)),
        mode="constant",
        constant_values=pad_value,
    )


def estimate_wrist_size(labels: np.ndarray, metric: str = "bounding_box_width") -> dict:
    pts = np.argwhere(labels > 0)
    if pts.size == 0:
        raise ValueError("No anatomy found in the label map.")
    r_min, c_min = pts.min(axis=0)
    r_max, c_max = pts.max(axis=0)
    bbox_height_px = int(r_max - r_min + 1)
    bbox_width_px = int(c_max - c_min + 1)
    size_pixels = bbox_width_px if metric == "bounding_box_width" else bbox_height_px
    return {
        "metric": metric,
        "size_pixels": int(size_pixels),
        "bbox_width_px": bbox_width_px,
        "bbox_height_px": bbox_height_px,
        "row_min": int(r_min),
        "row_max": int(r_max),
        "col_min": int(c_min),
        "col_max": int(c_max),
    }


def rescale_labels_to_target_size(
    labels: np.ndarray,
    target_size_mm: float,
    dx_m: float,
    metric: str = "bounding_box_width",
) -> tuple:
    size_info_before = estimate_wrist_size(labels, metric=metric)
    current_size_mm = size_info_before["size_pixels"] * dx_m * 1e3
    scale_factor = target_size_mm / current_size_mm
    labels_scaled = nd_zoom(
        labels,
        zoom=(scale_factor, scale_factor),
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
    ).astype(np.int32)
    size_info_after = estimate_wrist_size(labels_scaled, metric=metric)
    return (
        labels_scaled,
        {
            "scale_factor": scale_factor,
            "scaled_size_mm": size_info_after["size_pixels"] * dx_m * 1e3,
            "original_shape": tuple(labels.shape),
            "scaled_shape": tuple(labels_scaled.shape),
        },
        size_info_before,
        size_info_after,
    )


def place_labels_in_reference_frame(
    labels: np.ndarray,
    canvas_nx: int,
    canvas_ny: int,
    reference_top_margin_px: int,
    reference_center_col: int = None,
    reference_center_row: int = None,
    bone_label_id: int = 1,
    reference_bone_center_row: int = None,
) -> tuple:
    pts = np.argwhere(labels > 0)
    r_min, c_min = pts.min(axis=0)
    r_max, c_max = pts.max(axis=0)
    centroid_row, centroid_col = pts.mean(axis=0)
    if reference_center_col is None:
        reference_center_col = canvas_ny // 2
    row_shift = (
        int(round(reference_top_margin_px - r_min))
        if reference_center_row is None
        else int(round(reference_center_row - centroid_row))
    )
    col_shift = int(round(reference_center_col - centroid_col))
    bone_pts = np.argwhere(labels == bone_label_id)
    bone_centroid_row_before = float(bone_pts[:, 0].mean()) if bone_pts.size > 0 else None
    if reference_bone_center_row is not None and bone_pts.size > 0:
        shifted_bone_centroid_row = bone_centroid_row_before + row_shift
        row_shift += int(round(reference_bone_center_row - shifted_bone_centroid_row))
    shifted_pts = pts + np.array([row_shift, col_shift], dtype=int)
    shifted_r_min, shifted_c_min = shifted_pts.min(axis=0)
    shifted_r_max, shifted_c_max = shifted_pts.max(axis=0)
    if (
        shifted_r_min < 0
        or shifted_c_min < 0
        or shifted_r_max >= canvas_nx
        or shifted_c_max >= canvas_ny
    ):
        raise ValueError("Scaled wrist does not fit inside the reference canvas.")
    placed = np.zeros((canvas_nx, canvas_ny), dtype=np.int32)
    placed[shifted_pts[:, 0], shifted_pts[:, 1]] = labels[pts[:, 0], pts[:, 1]]
    return (
        placed,
        {
            "canvas_shape": (canvas_nx, canvas_ny),
            "row_shift": int(row_shift),
            "col_shift": int(col_shift),
            "reference_top_margin_px": int(reference_top_margin_px),
            "placed_row_min": int(shifted_r_min),
            "placed_row_max": int(shifted_r_max),
            "placed_col_min": int(shifted_c_min),
            "placed_col_max": int(shifted_c_max),
        },
    )


def ensure_reference_canvas_size(
    labels: np.ndarray,
    canvas_nx: int,
    canvas_ny: int,
    reference_top_margin_px: int,
    pml_size: int,
    min_pml_clearance_pixels: int,
    fixed_probe_offset_px: int,
    n_clusters: int,
    elements_per_cluster: int,
    element_width: int,
    intra_cluster_pitch_px: int,
    cluster_edge_overhang_px: int,
) -> tuple:
    size_info = estimate_wrist_size(labels, metric=WRIST_SIZE_METRIC)
    bbox_width_px, bbox_height_px = size_info["bbox_width_px"], size_info["bbox_height_px"]
    cluster_width = (elements_per_cluster - 1) * intra_cluster_pitch_px + element_width
    desired_array_width_px = max(bbox_width_px + 2 * cluster_edge_overhang_px, n_clusters * cluster_width)
    side_guard_px = pml_size + min_pml_clearance_pixels + 4
    top_guard_px = max(reference_top_margin_px, pml_size + min_pml_clearance_pixels + fixed_probe_offset_px)
    bottom_guard_px = pml_size + min_pml_clearance_pixels + 4
    return (
        max(canvas_nx, top_guard_px + bbox_height_px + bottom_guard_px),
        max(canvas_ny, desired_array_width_px + 2 * side_guard_px),
        top_guard_px,
        desired_array_width_px,
    )


def build_single_source_mask(element_patch: np.ndarray) -> np.ndarray:
    return element_patch.astype(np.uint8)


def average_element_signals(p_full, element_patches, sensor_mask):
    nt = p_full.shape[1]
    n_elements = len(element_patches)
    pixel_indices = np.argwhere(sensor_mask > 0)
    p_elements = np.zeros((n_elements, nt), dtype=np.float64)
    for el_idx, el_mask in enumerate(element_patches):
        rows_in_output = [i for i, (r, c) in enumerate(pixel_indices) if el_mask[r, c]]
        p_elements[el_idx] = p_full[rows_in_output, :].mean(axis=0)
    return p_elements


# ============================================================
# SMARTWATCH-SPECIFIC HELPERS (verbatim from standalone smartwatch_simulation.py)
# ============================================================

def build_smartwatch_transducer_geometry(
    labels: np.ndarray,
    offset_rows: int,
    pml_size: int,
    min_pml_clearance_pixels: int,
    transmitter_width: int,
    transmitter_height: int,
    tx_rx_gap_px: int,
    n_receivers: int,
    receiver_width: int,
    receiver_height: int,
    receiver_pitch_px: int,
    jitter_row: int = 0,
    jitter_col: int = 0
):
    """
    Build a smartwatch-like compact probe with one larger transmitter and a few
    nearby larger receivers on the same probe side.

    The transmitter is centered laterally over the wrist. Receivers are placed
    in a compact row just below it so the returned echoes stay strong while the
    acquisition remains close to a smartwatch form factor.
    """
    anatomy_mask = labels > 0
    pts = np.argwhere(anatomy_mask)
    if pts.size == 0:
        raise ValueError("No anatomy found in the label map (all labels == 0).")

    row_top = int(pts[:, 0].min())
    col_left = int(pts[:, 1].min())
    col_right = int(pts[:, 1].max())
    col_center = 0.5 * (col_left + col_right) + jitter_col

    tx_row_start = row_top - offset_rows + jitter_row
    tx_row_start = max(pml_size + min_pml_clearance_pixels, tx_row_start)
    tx_row_end = tx_row_start + transmitter_height - 1
    rx_row_start = tx_row_end + 1 + tx_rx_gap_px
    rx_row_end = rx_row_start + receiver_height - 1

    min_probe_row = pml_size + min_pml_clearance_pixels
    if tx_row_start < min_probe_row:
        raise ValueError(
            f"Smartwatch transmitter row {tx_row_start} is too close to the PML. "
            f"Need tx_row_start >= {min_probe_row}."
        )

    tx_center_col = int(round(col_center))
    tx_col_start = tx_center_col - transmitter_width // 2
    tx_col_end = tx_col_start + transmitter_width - 1

    usable_left = pml_size + min_pml_clearance_pixels
    usable_right = labels.shape[1] - (pml_size + min_pml_clearance_pixels) - 1

    if tx_col_start < usable_left or tx_col_end > usable_right:
        raise ValueError(
            "Smartwatch transmitter does not fit within the usable lateral bounds. "
            "Increase canvas width or reduce TRANSMITTER_WIDTH."
        )

    if np.any(labels[tx_row_start:tx_row_end + 1, tx_col_start:tx_col_end + 1] != 0):
        raise ValueError("Smartwatch transmitter overlaps anatomy.")

    total_rx_width = (n_receivers - 1) * receiver_pitch_px + receiver_width
    rx_array_left = int(round(col_center - 0.5 * (total_rx_width - 1)))
    rx_array_right = rx_array_left + total_rx_width - 1
    if rx_array_left < usable_left or rx_array_right > usable_right:
        raise ValueError(
            "Smartwatch receiver row does not fit within the usable lateral bounds. "
            "Increase canvas width or reduce N_RECEIVERS / RECEIVER_PITCH_PX."
        )

    tx_patch = np.zeros(labels.shape, dtype=bool)
    tx_patch[tx_row_start:tx_row_end + 1, tx_col_start:tx_col_end + 1] = True

    sensor_mask = np.zeros(labels.shape, dtype=np.uint8)
    element_patches = []
    element_centers = []
    cluster_indices = []
    local_position_indices = []
    element_metadata = []

    for rx_idx in range(n_receivers):
        center_col = int(
            round(rx_array_left + 0.5 * (receiver_width - 1) + rx_idx * receiver_pitch_px)
        )
        c_start = center_col - receiver_width // 2
        c_end = c_start + receiver_width - 1

        if c_start < usable_left or c_end > usable_right:
            raise ValueError(f"Receiver {rx_idx} is too close to the side PML.")
        if np.any(labels[rx_row_start:rx_row_end + 1, c_start:c_end + 1] != 0):
            raise ValueError(f"Receiver {rx_idx} overlaps anatomy.")

        rx_mask = np.zeros(labels.shape, dtype=bool)
        rx_mask[rx_row_start:rx_row_end + 1, c_start:c_end + 1] = True
        sensor_mask[rx_mask] = 1
        element_patches.append(rx_mask)

        center_row = rx_row_start + receiver_height // 2
        element_centers.append([center_row, center_col])
        cluster_indices.append(0)
        local_position_indices.append(rx_idx)
        element_metadata.append(
            {
                "global_element_index": rx_idx,
                "cluster_index": 0,
                "local_position_index": rx_idx,
                "row": int(center_row),
                "col": int(center_col),
                "row_start": int(rx_row_start),
                "row_end": int(rx_row_end),
                "col_start": int(c_start),
                "col_end": int(c_end),
            }
        )

    tx_metadata = {
        "row": int(tx_row_start + transmitter_height // 2),
        "col": int(tx_center_col),
        "row_start": int(tx_row_start),
        "row_end": int(tx_row_end),
        "col_start": int(tx_col_start),
        "col_end": int(tx_col_end),
        "width": int(transmitter_width),
        "height": int(transmitter_height),
    }

    return (
        tx_patch,
        tx_metadata,
        sensor_mask,
        element_patches,
        np.asarray(element_centers, dtype=int),
        np.asarray(cluster_indices, dtype=int),
        np.asarray(local_position_indices, dtype=int),
        int(tx_row_start),
        element_metadata,
    )


def compute_early_time_cutoff(
    labels: np.ndarray,
    probe_row: int,
    dx_m: float,
    dt_s: float,
    propagation_speed_m_per_s: float,
    safety_margin_px: int = 0,
    tx_metadata: dict = None,
    rx_metadata: list = None,
    direct_wave_margin_px: int = 0,
    burst_len_samples: int = 0,
    extra_samples: int = 0,
) -> dict:
    """
    Estimate when the first wrist echo could physically return to the probe.

    The cutoff is set conservatively using the larger of:
      - the first possible wrist round-trip echo time
      - the earliest direct Tx-to-Rx travel time plus a buffer

    This helps suppress both probe turn-on artefacts and direct-wave
    contamination in compact Tx/Rx layouts.
    """
    if dt_s <= 0:
        raise ValueError(f"dt_s must be > 0, got {dt_s}.")
    if propagation_speed_m_per_s <= 0:
        raise ValueError(
            f"propagation_speed_m_per_s must be > 0, got {propagation_speed_m_per_s}."
        )

    anatomy_pts = np.argwhere(labels > 0)
    if anatomy_pts.size == 0:
        raise ValueError("No anatomy found in the label map (all labels == 0).")

    top_anatomy_row = int(anatomy_pts[:, 0].min())
    probe_to_anatomy_depth_px = max(0, top_anatomy_row - probe_row)
    total_depth_px = probe_to_anatomy_depth_px + max(0, int(safety_margin_px))
    round_trip_distance_m = 2.0 * total_depth_px * dx_m
    anatomy_cutoff_time_s = round_trip_distance_m / propagation_speed_m_per_s

    direct_path_distance_m = None
    direct_cutoff_time_s = None
    if tx_metadata is not None and rx_metadata:
        tx_r0 = tx_metadata["row_start"]
        tx_r1 = tx_metadata["row_end"]
        tx_c0 = tx_metadata["col_start"]
        tx_c1 = tx_metadata["col_end"]

        min_direct_distance_px = None
        for rx_meta in rx_metadata:
            rx_r0 = rx_meta["row_start"]
            rx_r1 = rx_meta["row_end"]
            rx_c0 = rx_meta["col_start"]
            rx_c1 = rx_meta["col_end"]

            row_gap_px = max(0, max(rx_r0 - tx_r1 - 1, tx_r0 - rx_r1 - 1))
            col_gap_px = max(0, max(rx_c0 - tx_c1 - 1, tx_c0 - rx_c1 - 1))
            pixel_distance = np.sqrt(row_gap_px**2 + col_gap_px**2)
            if min_direct_distance_px is None or pixel_distance < min_direct_distance_px:
                min_direct_distance_px = float(pixel_distance)

        if min_direct_distance_px is not None:
            direct_path_distance_m = (
                min_direct_distance_px + max(0, int(direct_wave_margin_px))
            ) * dx_m
            direct_cutoff_time_s = (
                direct_path_distance_m / propagation_speed_m_per_s
                + max(0, int(burst_len_samples)) * dt_s
            )

    cutoff_time_candidates = [anatomy_cutoff_time_s]
    if direct_cutoff_time_s is not None:
        cutoff_time_candidates.append(direct_cutoff_time_s)
    cutoff_time_s = max(cutoff_time_candidates)
    cutoff_samples = int(np.ceil(cutoff_time_s / dt_s)) + max(0, int(extra_samples))

    return {
        "top_anatomy_row": top_anatomy_row,
        "probe_row": int(probe_row),
        "probe_to_anatomy_depth_px": int(probe_to_anatomy_depth_px),
        "safety_margin_px": int(max(0, int(safety_margin_px))),
        "total_depth_px": int(total_depth_px),
        "round_trip_distance_m": float(round_trip_distance_m),
        "anatomy_cutoff_time_s": float(anatomy_cutoff_time_s),
        "direct_wave_margin_px": int(max(0, int(direct_wave_margin_px))),
        "direct_path_distance_m": None
        if direct_path_distance_m is None
        else float(direct_path_distance_m),
        "direct_cutoff_time_s": None
        if direct_cutoff_time_s is None
        else float(direct_cutoff_time_s),
        "burst_len_samples": int(max(0, int(burst_len_samples))),
        "extra_samples": int(max(0, int(extra_samples))),
        "cutoff_time_s": float(cutoff_time_s),
        "cutoff_samples": int(cutoff_samples),
    }


# ============================================================
# MAIN SIMULATION FUNCTION
# ============================================================

def run_smartwatch_simulation_on_file(
    input_file: str,
    execution_threads: int = 1,
    use_gpu: bool = False,
    gpu_id: int = None,
) -> np.ndarray:
    """Run the smartwatch simulation on one 2D segmentation.

    Returns:
        np.ndarray: shape (1, 4, Nt_cropped) — 1 Tx × 4 Rx × time
    """
    if not use_gpu:
        os.environ["KWAVE_FORCE_CPU"] = "1"
    else:
        os.environ.pop("KWAVE_FORCE_CPU", None)

    os.environ["OMP_NUM_THREADS"] = str(execution_threads)
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    raw_labels = load_2d_labels(input_file)
    labels = force_2d_labels(raw_labels)

    labels, _ = register_labels_to_canonical_axis(labels, canonical=CANONICAL_AXIS)
    if ENABLE_BONE_SIDE_CORRECTION:
        labels, _ = enforce_bones_in_lower_half(labels, bone_label_id=BONE_LABEL_ID)

    array_geometry_info = compute_required_wrist_pixels_for_array_geometry(
        N_CLUSTERS,
        ELEMENTS_PER_CLUSTER,
        ELEMENT_WIDTH,
        INTRA_CLUSTER_PITCH_PX,
        CLUSTER_EDGE_OVERHANG_PX,
        MIN_INTER_CLUSTER_EDGE_GAP_PX,
    )
    grid_design_info = choose_grid_spacing_for_geometry(
        C_REF_GRID,
        TONE_FREQ,
        PPW,
        TARGET_WRIST_WIDTH_MM,
        array_geometry_info["required_wrist_width_px"],
    )
    dx_design = grid_design_info["dx"]
    dy_design = grid_design_info["dy"]
    lam_min_design = grid_design_info["lam_min"]
    min_pml_clearance_m = MIN_PML_TRANSDUCER_WAVELENGTHS * lam_min_design
    min_pml_clearance_pixels = int(np.ceil(min_pml_clearance_m / dx_design))

    labels_scaled, _, _, _ = rescale_labels_to_target_size(
        labels, TARGET_WRIST_WIDTH_MM, dx_design, metric=WRIST_SIZE_METRIC
    )

    canvas_nx_used, canvas_ny_used, top_margin_used, _ = ensure_reference_canvas_size(
        labels_scaled,
        REFERENCE_CANVAS_NX,
        REFERENCE_CANVAS_NY,
        REFERENCE_TOP_MARGIN_PX,
        PML_SIZE,
        min_pml_clearance_pixels,
        FIXED_PROBE_OFFSET_PX,
        N_CLUSTERS,
        ELEMENTS_PER_CLUSTER,
        ELEMENT_WIDTH,
        INTRA_CLUSTER_PITCH_PX,
        CLUSTER_EDGE_OVERHANG_PX,
    )

    labels_positioned, placement_info = place_labels_in_reference_frame(
        labels_scaled,
        canvas_nx_used,
        canvas_ny_used,
        top_margin_used,
        REFERENCE_CENTER_COL,
        REFERENCE_CENTER_ROW,
        BONE_LABEL_ID,
        REFERENCE_BONE_CENTER_ROW if ENABLE_BONE_DEPTH_ALIGNMENT else None,
    )

    # Bone alignment may shift the anatomy upward past the safe top margin.
    # Pad extra rows at the top to guarantee transmitter clearance from the PML.
    min_row_top_needed = PML_SIZE + min_pml_clearance_pixels + FIXED_PROBE_OFFSET_PX
    row_top_placed = placement_info["placed_row_min"]
    if row_top_placed < min_row_top_needed:
        extra_pad = min_row_top_needed - row_top_placed
        labels_positioned = np.pad(labels_positioned, ((extra_pad, 0), (0, 0)), constant_values=0)

    sound_speed_map, density_map, alpha_coeff_map, _ = build_property_maps(labels_positioned)
    bg = labels_positioned == 0
    sound_speed_map[bg] = BG_C
    density_map[bg] = BG_RHO
    alpha_coeff_map[bg] = BG_ALPHA

    labels_padded = pad_to_fft_friendly(labels_positioned, pad_value=0)
    sound_speed_padded = pad_to_fft_friendly(sound_speed_map, pad_value=BG_C)
    density_padded = pad_to_fft_friendly(density_map, pad_value=BG_RHO)
    alpha_coeff_padded = pad_to_fft_friendly(alpha_coeff_map, pad_value=BG_ALPHA)
    labels = labels_padded

    dx, dy = dx_design, dy_design
    nx, ny = labels.shape
    kgrid = kWaveGrid([nx, ny], [dx, dy])

    medium = kWaveMedium(
        sound_speed=sound_speed_padded,
        density=density_padded,
        alpha_coeff=alpha_coeff_padded,
        alpha_power=ALPHA_POWER,
    )

    rng = np.random.default_rng()
    jitter_row = rng.integers(-4, 5) # [-4, 4] pixels
    jitter_col = rng.integers(-2, 3) # [-2, 2] pixels

    (
        tx_patch,
        tx_metadata,
        sensor_mask,
        element_patches,
        _,
        _,
        _,
        probe_row,
        element_metadata,
    ) = build_smartwatch_transducer_geometry(
        labels,
        FIXED_PROBE_OFFSET_PX,
        PML_SIZE,
        min_pml_clearance_pixels,
        TRANSMITTER_WIDTH,
        TRANSMITTER_HEIGHT,
        TX_RX_GAP_PX,
        N_RECEIVERS,
        RECEIVER_WIDTH,
        RECEIVER_HEIGHT,
        RECEIVER_PITCH_PX,
        jitter_row=jitter_row,
        jitter_col=jitter_col
    )

    sensor = kSensor(mask=sensor_mask)
    sensor.record = ["p"]
    n_sensor_pixels = int(np.count_nonzero(sensor_mask))

    simulation_options = SimulationOptions(
        pml_inside=PML_INSIDE,
        pml_size=PML_SIZE,
        pml_alpha=PML_ALPHA,
        save_to_disk=SAVE_TO_DISK,
    )
    execution_options = SimulationExecutionOptions(
        is_gpu_simulation=use_gpu,
        device_num=gpu_id,
    )
    execution_options.num_threads = execution_threads

    # Copy kwave binary to local /tmp/ to prevent ETXTBSY when multiple SLURM jobs
    # execute the same binary concurrently from a shared network filesystem.
    _orig_binary = execution_options.binary_path
    _tmp_binary_path = Path(tempfile.gettempdir()) / f"kwave_bin_{uuid.uuid4().hex}"
    shutil.copy2(_orig_binary, _tmp_binary_path)
    os.chmod(_tmp_binary_path, 0o755)
    execution_options.binary_path = _tmp_binary_path

    kgrid.makeTime(sound_speed_padded, cfl=CFL)
    nt = int(np.asarray(kgrid.t_array).size)
    dt = float(kgrid.dt)
    fs = 1.0 / dt

    burst_raw = gaussian_pulse(fs, TONE_FREQ, TONE_CYCLES)
    burst_len = int(burst_raw.size)
    p_t = np.zeros(nt, dtype=np.float32)
    p_t[:burst_len] = SOURCE_PA * burst_raw

    all_rf_data = []
    local_tmp = os.path.join(os.getcwd(), "output", "tmp_h5")
    os.makedirs(local_tmp, exist_ok=True)

    unique_id = uuid.uuid4().hex
    input_fname = os.path.join(local_tmp, f"kwave_in_sw_{unique_id}.h5")
    output_fname = os.path.join(local_tmp, f"kwave_out_sw_{unique_id}.h5")

    simulation_options.input_filename = input_fname
    simulation_options.output_filename = output_fname

    source_mask = build_single_source_mask(tx_patch)
    
    apo_1d = apply_apodization(tx_patch, window_type="hann")
    
    source = kSource()
    source.p_mask = source_mask
    source.p = (apo_1d[:, np.newaxis] * p_t[np.newaxis, :]).astype(np.float32)

    try:
        try:
            sim_data = kspaceFirstOrder2D(
                kgrid=kgrid,
                medium=medium,
                source=source,
                sensor=sensor,
                simulation_options=simulation_options,
                execution_options=execution_options,
            )
            p_raw = np.asarray(
                sim_data["p"] if isinstance(sim_data, dict) else sim_data, dtype=np.float64
            )
            if p_raw.shape[0] == nt and p_raw.shape[1] == n_sensor_pixels:
                p_raw = p_raw.T
            p_raw = np.nan_to_num(p_raw, nan=0.0, posinf=0.0, neginf=0.0)
            p_elements = average_element_signals(p_raw, element_patches, sensor_mask)
            all_rf_data.append(p_elements)
        finally:
            if os.path.exists(input_fname):
                try:
                    os.remove(input_fname)
                except Exception:
                    pass
            if os.path.exists(output_fname):
                try:
                    os.remove(output_fname)
                except Exception:
                    pass
    finally:
        try:
            _tmp_binary_path.unlink(missing_ok=True)
        except Exception:
            pass

    if APPLY_EARLY_TIME_CUTOFF:
        cutoff_info = compute_early_time_cutoff(
            labels,
            probe_row,
            dx,
            dt,
            BG_C,
            safety_margin_px=EARLY_TIME_CUTOFF_MARGIN_PX,
            tx_metadata=tx_metadata,
            rx_metadata=element_metadata,
            direct_wave_margin_px=DIRECT_WAVE_CUTOFF_MARGIN_PX,
            burst_len_samples=burst_len,
            extra_samples=EARLY_TIME_CUTOFF_EXTRA_SAMPLES,
        )
        cutoff_samples = min(int(cutoff_info["cutoff_samples"]), nt - 1)
        all_rf_data = [p[:, cutoff_samples:].copy() for p in all_rf_data]

    rf_volume = np.stack(all_rf_data, axis=0)
    return rf_volume
