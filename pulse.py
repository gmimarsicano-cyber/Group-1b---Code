import numpy as np

def gaussian_pulse(fs, freq, cycles):
    """
    Creates a Gaussian-modulated sine wave pulse.
    fs: sampling frequency (Hz)
    freq: center frequency (Hz)
    cycles: number of cycles in the pulse
    """
    t_length = cycles / freq
    t = np.arange(-t_length/2, t_length/2, 1/fs)
    # Gaussian envelope where 6 sigma covers the entire length
    sigma = t_length / 6 
    envelope = np.exp(- (t**2) / (2 * sigma**2))
    pulse = envelope * np.sin(2 * np.pi * freq * t)
    return pulse.astype(np.float32)

def apply_apodization(element_patch, window_type="hann"):
    """
    Creates a 1D apodization array corresponding to the active pixels in element_patch.
    This array is formatted to multiply correctly with the source.p time array.
    """
    pts = np.argwhere(element_patch)
    if pts.size == 0:
        return np.array([], dtype=np.float32)
        
    r_min, c_min = pts.min(axis=0)
    r_max, c_max = pts.max(axis=0)
    width = c_max - c_min + 1
    
    apo_2d = np.zeros_like(element_patch, dtype=np.float32)
    
    if window_type == "hann":
        window = np.hanning(width).astype(np.float32)
    elif window_type == "sinc":
        x = np.linspace(-np.pi, np.pi, width)
        # Avoid division by zero warning, sinc handles it
        window = np.sinc(x / np.pi).astype(np.float32)
    else:
        window = np.ones(width, dtype=np.float32)
        
    for c_idx in range(c_min, c_max + 1):
        apo_2d[r_min : r_max + 1, c_idx] = window[c_idx - c_min]
        
    # Extract just the values where the patch is active to match k-Wave flattened format
    return apo_2d[element_patch > 0]
