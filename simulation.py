import os
import uuid
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
from kwave.utils.signals import tone_burst
from src.kwave.pulse import gaussian_pulse, apply_apodization

from src.kwave.tissue_mapping_2d import load_2d_labels, build_property_maps

# ============================================================
# USER SETTINGS / CONSTANTS
# ============================================================
TONE_FREQ   = 5e6
TONE_CYCLES = 3
SOURCE_PA   = 2e5

N_SENSORS      = 8
SENSOR_OFFSET  = 30
ELEMENT_WIDTH  = 5
ELEMENT_HEIGHT = 1

PPW        = 4
C_REF_GRID = 1475.0

BG_C        = 1500.0
BG_RHO      = 1000.0
BG_ALPHA    = 0.002
ALPHA_POWER = 0.8

CANONICAL_AXIS           = "horizontal"
TARGET_WRIST_WIDTH_MM    = 55
WRIST_SIZE_METRIC        = "bounding_box_width"
REFERENCE_CANVAS_NX      = 64   # lower bound only; ensure_reference_canvas_size computes the real size
REFERENCE_CANVAS_NY      = 64   # lower bound only
REFERENCE_TOP_MARGIN_PX  = 80   # 80px × dx = ~6mm water coupling above skin at PPW=4
ENABLE_BONE_SIDE_CORRECTION = True
BONE_LABEL_ID            = 1
ENABLE_BONE_DEPTH_ALIGNMENT = False
REFERENCE_BONE_CENTER_ROW = 1850  # dormant; kept for reference

PML_INSIDE = True
PML_SIZE   = 20
PML_ALPHA  = 2.0
MIN_PML_TRANSDUCER_WAVELENGTHS = 2.0

APPLY_EARLY_TIME_CUTOFF    = True
EARLY_TIME_CUTOFF_MARGIN_PX = 5

ENABLE_SOS_JITTER = True
SOS_JITTER_PCT    = 0.03

CFL          = 0.3
SAVE_TO_DISK = True

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def force_2d_labels(raw: np.ndarray) -> np.ndarray:
    arr = np.squeeze(raw).astype(np.int32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D slice, got {arr.ndim}D after squeezing.")
    return arr

def register_labels_to_canonical_axis(labels: np.ndarray, canonical: str = "vertical") -> tuple:
    pts = np.argwhere(labels > 0).astype(np.float64)
    if pts.shape[0] < 2:
        return labels, 0.0
    pts -= pts.mean(axis=0)
    cov = np.cov(pts.T)
    _, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, -1]
    angle_from_horizontal = np.degrees(np.arctan2(principal[0], principal[1]))
    target_angle = 90.0 if canonical == "vertical" else 0.0
    angle_deg = target_angle - angle_from_horizontal
    if angle_deg > 90:
        angle_deg -= 180.0
    elif angle_deg < -90:
        angle_deg += 180.0
    labels_reg = nd_rotate(labels, angle=angle_deg, axes=(0, 1), reshape=True, order=0, cval=0, prefilter=False)
    return labels_reg.astype(np.int32), angle_deg

def enforce_bones_in_lower_half(labels: np.ndarray, bone_label_id: int = 1) -> tuple:
    anatomy_pts = np.argwhere(labels > 0)
    bone_pts = np.argwhere(labels == bone_label_id)
    if anatomy_pts.size == 0 or bone_pts.size == 0:
        return labels, {}
    anatomy_centroid_row = float(anatomy_pts[:, 0].mean())
    bone_centroid_row = float(bone_pts[:, 0].mean())
    if bone_centroid_row < anatomy_centroid_row:
        return np.rot90(labels, 2).astype(np.int32), {"rotation_applied_deg": 180}
    return labels, {"rotation_applied_deg": 0}

def compute_dx_from_ppw(c_min: float, f_max: float, ppw: int):
    lam_min = c_min / f_max
    dx = lam_min / ppw
    return dx, dx, lam_min

def estimate_wrist_size(labels: np.ndarray, metric: str = "bounding_box_width") -> dict:
    pts = np.argwhere(labels > 0)
    if pts.size == 0:
        raise ValueError("No anatomy found in the label map.")
    r_min, c_min = pts.min(axis=0)
    r_max, c_max = pts.max(axis=0)
    bbox_width_px  = int(c_max - c_min + 1)
    bbox_height_px = int(r_max - r_min + 1)
    size_pixels = bbox_width_px if metric == "bounding_box_width" else bbox_height_px
    return {"metric": metric, "size_pixels": int(size_pixels),
            "bbox_width_px": bbox_width_px, "bbox_height_px": bbox_height_px,
            "row_min": int(r_min), "row_max": int(r_max),
            "col_min": int(c_min), "col_max": int(c_max)}

def rescale_labels_to_target_size(labels: np.ndarray, target_size_mm: float, dx_m: float,
                                   metric: str = "bounding_box_width") -> tuple:
    size_info_before = estimate_wrist_size(labels, metric=metric)
    current_size_mm = size_info_before["size_pixels"] * dx_m * 1e3
    scale_factor = target_size_mm / current_size_mm
    labels_scaled = nd_zoom(labels, zoom=(scale_factor, scale_factor), order=0,
                            mode="constant", cval=0, prefilter=False).astype(np.int32)
    size_info_after = estimate_wrist_size(labels_scaled, metric=metric)
    return labels_scaled, {"scale_factor": scale_factor,
                           "scaled_size_mm": size_info_after["size_pixels"] * dx_m * 1e3,
                           "original_shape": tuple(labels.shape),
                           "scaled_shape": tuple(labels_scaled.shape)}, size_info_before, size_info_after

def ensure_reference_canvas_size(labels: np.ndarray, canvas_nx: int, canvas_ny: int, reference_top_margin_px: int, pml_size: int, min_pml_clearance_pixels: int, fixed_probe_offset_px: int) -> tuple:
    size_info = estimate_wrist_size(labels, metric=WRIST_SIZE_METRIC)
    bbox_width_px, bbox_height_px = size_info["bbox_width_px"], size_info["bbox_height_px"]
    
    side_guard_px = pml_size + min_pml_clearance_pixels + 4
    top_guard_px = max(reference_top_margin_px, pml_size + min_pml_clearance_pixels + fixed_probe_offset_px)
    bottom_guard_px = pml_size + min_pml_clearance_pixels + 4
    
    return max(canvas_nx, top_guard_px + bbox_height_px + bottom_guard_px), max(canvas_ny, bbox_width_px + 2 * side_guard_px), top_guard_px

def place_labels_in_reference_frame(labels: np.ndarray, canvas_nx: int, canvas_ny: int,
                                     reference_top_margin_px: int, reference_center_col: int = None,
                                     bone_label_id: int = 1,
                                     reference_bone_center_row: int = None) -> tuple:
    pts = np.argwhere(labels > 0)
    r_min, _ = pts.min(axis=0)
    _, centroid_col = pts.mean(axis=0)
    if reference_center_col is None:
        reference_center_col = canvas_ny // 2
    row_shift = int(round(reference_top_margin_px - r_min))
    col_shift = int(round(reference_center_col - centroid_col))
    bone_pts = np.argwhere(labels == bone_label_id)
    if reference_bone_center_row is not None and bone_pts.size > 0:
        bone_centroid_row_before = float(bone_pts[:, 0].mean())
        shifted_bone_row = bone_centroid_row_before + row_shift
        row_shift += int(round(reference_bone_center_row - shifted_bone_row))
    shifted_pts = pts + np.array([row_shift, col_shift], dtype=int)
    shifted_r_min, shifted_c_min = shifted_pts.min(axis=0)
    shifted_r_max, shifted_c_max = shifted_pts.max(axis=0)
    if shifted_r_min < 0 or shifted_c_min < 0 or shifted_r_max >= canvas_nx or shifted_c_max >= canvas_ny:
        raise ValueError("Scaled wrist does not fit inside the reference canvas.")
    placed = np.zeros((canvas_nx, canvas_ny), dtype=np.int32)
    placed[shifted_pts[:, 0], shifted_pts[:, 1]] = labels[pts[:, 0], pts[:, 1]]
    return placed, {"canvas_shape": (canvas_nx, canvas_ny), "row_shift": int(row_shift),
                    "col_shift": int(col_shift), "placed_row_min": int(shifted_r_min),
                    "placed_row_max": int(shifted_r_max)}

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
    Nx_target = next_smooth(arr.shape[0])
    Ny_target = next_smooth(arr.shape[1])
    pad_x = Nx_target - arr.shape[0]
    pad_y = Ny_target - arr.shape[1]
    px0, px1 = pad_x // 2, pad_x - pad_x // 2
    py0, py1 = pad_y // 2, pad_y - pad_y // 2
    return np.pad(arr, pad_width=((px0, px1), (py0, py1)), mode="constant", constant_values=pad_value)

def build_transducer_array(labels, n_sensors, offset_rows, pml_size, element_width, element_height, jitter_row=0, jitter_col=0):
    anatomy_mask = labels > 0
    pts = np.argwhere(anatomy_mask)
    if pts.size == 0:
        raise ValueError("No anatomy found in the label map.")
    row_top  = pts[:, 0].min()
    col_left = pts[:, 1].min()
    col_right = pts[:, 1].max()
    
    probe_row = row_top - offset_rows + jitter_row
    # Clamp probe_row to be strictly below the PML
    probe_row = max(pml_size, probe_row)
    probe_row_bottom = probe_row + element_height - 1
    
    if probe_row < pml_size:
        raise ValueError(f"Probe top row {probe_row} is inside PML (pml_size={pml_size}).")
    patch_rows = slice(probe_row, probe_row_bottom + 1)
    if np.any(labels[patch_rows, col_left:col_right + 1] != 0):
        raise ValueError("Probe rows overlap anatomy.")
    
    center_cols = np.round(np.linspace(col_left, col_right, n_sensors)).astype(int) + jitter_col
    half_w = element_width // 2
    used_cols = set()
    for cc in center_cols:
        patch_cols = set(range(cc - half_w, cc + (element_width - half_w)))
        if patch_cols & used_cols:
            raise ValueError("Element patches overlap.")
        used_cols |= patch_cols
    Nx, Ny = labels.shape
    sensor_mask = np.zeros((Nx, Ny), dtype=np.uint8)
    element_patches = []
    for cc in center_cols:
        c_start = max(0, cc - half_w)
        c_end = min(Ny - 1, cc + (element_width - half_w) - 1)
        if c_start < pml_size or c_end >= Ny - pml_size:
            raise ValueError("Element falls inside column PML.")
        el_mask = np.zeros((Nx, Ny), dtype=bool)
        el_mask[probe_row: probe_row_bottom + 1, c_start: c_end + 1] = True
        sensor_mask[el_mask] = 1
        element_patches.append(el_mask)
    element_centers = np.array([[probe_row + element_height // 2, int(cc)] for cc in center_cols])
    return sensor_mask, element_patches, element_centers, probe_row

def build_single_source_mask(element_patch):
    return element_patch.astype(np.uint8)

def compute_early_time_cutoff(labels, probe_row, dx_m, dt_s, propagation_speed_m_per_s, safety_margin_px=0):
    anatomy_pts = np.argwhere(labels > 0)
    top_anatomy_row = int(anatomy_pts[:, 0].min())
    probe_to_anatomy_depth_px = max(0, top_anatomy_row - probe_row)
    total_depth_px = probe_to_anatomy_depth_px + max(0, int(safety_margin_px))
    round_trip_distance_m = 2.0 * total_depth_px * dx_m
    cutoff_time_s = round_trip_distance_m / propagation_speed_m_per_s
    return {"cutoff_samples": int(np.ceil(cutoff_time_s / dt_s))}

def average_element_signals(p_full, element_patches, sensor_mask):
    Nt = p_full.shape[1]
    n_elements = len(element_patches)
    pixel_indices = np.argwhere(sensor_mask > 0)
    p_elements = np.zeros((n_elements, Nt), dtype=np.float64)
    for el_idx, el_mask in enumerate(element_patches):
        rows_in_output = [i for i, (r, c) in enumerate(pixel_indices) if el_mask[r, c]]
        p_elements[el_idx] = p_full[rows_in_output, :].mean(axis=0)
    return p_elements


def run_simulation_on_file(input_file: str, execution_threads: int = 1, use_gpu: bool = False, gpu_id: int = None):
    """
    Runs the k-Wave simulation on a single segmented input file and returns the generated rf_volume.
    """
    if not use_gpu:
        os.environ["KWAVE_FORCE_CPU"] = "1"
    else:
        os.environ.pop("KWAVE_FORCE_CPU", None)

    os.environ["OMP_NUM_THREADS"] = str(execution_threads)
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    # 1. Load and prep labels
    raw_labels = load_2d_labels(input_file)
    labels = force_2d_labels(raw_labels)

    # 2. Orient anatomy: align long axis, then ensure bones are in lower half
    labels, _ = register_labels_to_canonical_axis(labels, canonical=CANONICAL_AXIS)
    if ENABLE_BONE_SIDE_CORRECTION:
        labels, _ = enforce_bones_in_lower_half(labels, bone_label_id=BONE_LABEL_ID)

    # 3. Compute grid spacing (PPW-based) and PML clearance in pixels
    dx, dy, lam_min = compute_dx_from_ppw(C_REF_GRID, TONE_FREQ, PPW)
    min_pml_clearance_pixels = int(np.ceil(MIN_PML_TRANSDUCER_WAVELENGTHS * lam_min / dx))
    # Effective probe offset: no closer to PML than MIN_PML_TRANSDUCER_WAVELENGTHS*λ
    effective_offset = max(SENSOR_OFFSET, PML_SIZE + min_pml_clearance_pixels)

    # 4. Scale anatomy to target wrist width and place in reference frame with bone alignment
    labels_scaled, _, _, _ = rescale_labels_to_target_size(labels, TARGET_WRIST_WIDTH_MM, dx, metric=WRIST_SIZE_METRIC)

    canvas_nx_used, canvas_ny_used, top_margin_used = ensure_reference_canvas_size(
        labels_scaled, REFERENCE_CANVAS_NX, REFERENCE_CANVAS_NY, REFERENCE_TOP_MARGIN_PX,
        PML_SIZE, min_pml_clearance_pixels, effective_offset
    )

    labels_placed, placement_info = place_labels_in_reference_frame(
        labels_scaled, canvas_nx_used, canvas_ny_used, top_margin_used,
        reference_center_col=None,
        bone_label_id=BONE_LABEL_ID,
        reference_bone_center_row=REFERENCE_BONE_CENTER_ROW if ENABLE_BONE_DEPTH_ALIGNMENT else None,
    )

    # If bone alignment pushes anatomy above the safe probe-clearance zone, pad at top
    min_row_top_needed = PML_SIZE + min_pml_clearance_pixels + effective_offset
    if placement_info["placed_row_min"] < min_row_top_needed:
        extra_pad = min_row_top_needed - placement_info["placed_row_min"]
        labels_placed = np.pad(labels_placed, ((extra_pad, 0), (0, 0)), constant_values=0)

    # 5. Build acoustic property maps and fill background
    sound_speed_map, density_map, alpha_coeff_map, _ = build_property_maps(labels_placed)

    if ENABLE_SOS_JITTER:
        rng = np.random.default_rng()
        jitter = 1.0 + rng.uniform(-SOS_JITTER_PCT, SOS_JITTER_PCT)
        sound_speed_map = sound_speed_map * jitter
        bg_c_local = BG_C * jitter
    else:
        bg_c_local = BG_C

    bg = (labels_placed == 0)
    sound_speed_map[bg] = bg_c_local
    density_map[bg]     = BG_RHO
    alpha_coeff_map[bg] = BG_ALPHA

    labels_padded      = pad_to_fft_friendly(labels_placed,    pad_value=0)
    sound_speed_padded = pad_to_fft_friendly(sound_speed_map,  pad_value=bg_c_local)
    density_padded     = pad_to_fft_friendly(density_map,      pad_value=BG_RHO)
    alpha_coeff_padded = pad_to_fft_friendly(alpha_coeff_map,  pad_value=BG_ALPHA)
    labels = labels_padded

    # 6. Grid
    Nx, Ny = labels.shape
    kgrid = kWaveGrid([Nx, Ny], [dx, dy])

    # 7. Medium
    medium = kWaveMedium(
        sound_speed=sound_speed_padded,
        density=density_padded,
        alpha_coeff=alpha_coeff_padded,
        alpha_power=ALPHA_POWER,
    )

    # 8. Transducer
    rng = np.random.default_rng()
    jitter_row = rng.integers(-4, 5) # [-4, 4] pixels
    jitter_col = rng.integers(-2, 3) # [-2, 2] pixels
    
    sensor_mask, element_patches, _, probe_row = build_transducer_array(
        labels=labels, n_sensors=N_SENSORS, offset_rows=effective_offset,
        pml_size=PML_SIZE, element_width=ELEMENT_WIDTH, element_height=ELEMENT_HEIGHT,
        jitter_row=jitter_row, jitter_col=jitter_col
    )
    sensor = kSensor(mask=sensor_mask)
    sensor.record = ["p"]
    n_sensor_pixels = int(np.count_nonzero(sensor_mask))

    # 9. Options and time array
    simulation_options = SimulationOptions(
        pml_inside=PML_INSIDE, pml_size=PML_SIZE,
        pml_alpha=PML_ALPHA, save_to_disk=SAVE_TO_DISK
    )
    execution_options = SimulationExecutionOptions(is_gpu_simulation=use_gpu, device_num=gpu_id)
    execution_options.num_threads = execution_threads

    kgrid.makeTime(sound_speed_padded, cfl=CFL)
    Nt = int(np.asarray(kgrid.t_array).size)
    dt = float(kgrid.dt)
    fs = 1.0 / dt

    burst_raw = gaussian_pulse(fs, TONE_FREQ, TONE_CYCLES)
    burst_len = burst_raw.size
    p_t = np.zeros(Nt, dtype=np.float32)
    p_t[:burst_len] = SOURCE_PA * burst_raw

    # 10. Acquisition loop
    all_rf_data = []
    local_tmp = os.path.join(os.getcwd(), "output", "tmp_h5")
    os.makedirs(local_tmp, exist_ok=True)

    for tx_idx, el_patch in enumerate(element_patches):
        unique_id = uuid.uuid4().hex
        input_fname  = os.path.join(local_tmp, f"kwave_in_{unique_id}.h5")
        output_fname = os.path.join(local_tmp, f"kwave_out_{unique_id}.h5")

        simulation_options.input_filename  = input_fname
        simulation_options.output_filename = output_fname

        source_mask = build_single_source_mask(el_patch)
        n_src = int(np.count_nonzero(source_mask))

        source = kSource()
        source.p_mask = source_mask
        
        # Apply element directivity apodization (hann window)
        apo_1d = apply_apodization(el_patch, window_type="hann")
        # source.p must have shape (n_source_pixels, Nt)
        source.p = (apo_1d[:, np.newaxis] * p_t[np.newaxis, :]).astype(np.float32)

        try:
            sim_data = kspaceFirstOrder2D(
                kgrid=kgrid, medium=medium, source=source, sensor=sensor,
                simulation_options=simulation_options,
                execution_options=execution_options
            )
            p_raw = np.asarray(
                sim_data["p"] if isinstance(sim_data, dict) else sim_data,
                dtype=np.float64
            )
            if p_raw.shape[0] == Nt and p_raw.shape[1] == n_sensor_pixels:
                p_raw = p_raw.T
            p_raw = np.nan_to_num(p_raw, nan=0.0, posinf=0.0, neginf=0.0)
            p_elements = average_element_signals(p_raw, element_patches, sensor_mask)
            all_rf_data.append(p_elements)
        finally:
            if os.path.exists(input_fname):
                try: os.remove(input_fname)
                except Exception: pass
            if os.path.exists(output_fname):
                try: os.remove(output_fname)
                except Exception: pass

    # 11. Trim early samples before first physical echo can arrive
    if APPLY_EARLY_TIME_CUTOFF:
        cutoff_info = compute_early_time_cutoff(labels, probe_row, dx, dt, bg_c_local, EARLY_TIME_CUTOFF_MARGIN_PX)
        cutoff_samples = min(cutoff_info["cutoff_samples"], Nt - 1)
        all_rf_data = [p[:, cutoff_samples:].copy() for p in all_rf_data]

    # 12. Return RF volume: shape (N_tx, N_rx, Nt_after_cutoff)
    rf_volume = np.stack(all_rf_data, axis=0)
    return rf_volume
