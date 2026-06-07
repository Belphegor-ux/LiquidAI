import numpy as np
from scipy import signal

def extract_band_power(data, fs, window_sec=2.0):
    """
    Extracts spectral band power features from EEG data.

    Parameters:
    -----------
    data : numpy.ndarray
        The EEG data. Can be 1D (samples) or 2D (channels, samples).
    fs : float
        Sampling frequency of the EEG data in Hz.
    window_sec : float
        Length of each window in seconds for Welch's method.

    Returns:
    --------
    dict
        A dictionary containing the absolute band powers for each frequency band.
        If data is 2D, the values will be 1D arrays of shape (channels,).
    """
    # Define standard EEG frequency bands
    eeg_bands = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 13),
        'Beta': (13, 30),
        'Gamma': (30, 100)
    }

    # Compute Power Spectral Density (PSD) using Welch's method
    # Calculate segment length in samples
    nperseg = int(window_sec * fs)
    
    # Adjust nperseg if data is shorter than the window length
    if data.ndim == 1 and nperseg > len(data):
        nperseg = len(data)
    elif data.ndim > 1 and nperseg > data.shape[-1]:
        nperseg = data.shape[-1]
    
    freqs, psd = signal.welch(data, fs=fs, nperseg=nperseg)
    
    # Calculate absolute power in each band
    band_powers = {}
    
    for band, (fmin, fmax) in eeg_bands.items():
        # Find indices corresponding to the frequency band
        idx_band = np.logical_and(freqs >= fmin, freqs <= fmax)
        
        # Integrate PSD over the frequency band using the trapezoidal rule
        if data.ndim == 1:
            power = np.trapz(psd[idx_band], freqs[idx_band])
        else:
            power = np.trapz(psd[:, idx_band], freqs[idx_band], axis=1)
            
        band_powers[band] = power

    return band_powers

if __name__ == "__main__":
    # Example usage with synthetic data
    fs = 250.0  # Sampling frequency in Hz
    duration = 10.0  # Duration in seconds
    t = np.arange(int(fs * duration)) / fs
    
    # Create a synthetic EEG signal with specific frequency components
    # 2 Hz (Delta), 6 Hz (Theta), 10 Hz (Alpha), 20 Hz (Beta), 40 Hz (Gamma)
    synthetic_eeg = (
        1.5 * np.sin(2 * np.pi * 2 * t) +
        1.0 * np.sin(2 * np.pi * 6 * t) +
        2.0 * np.sin(2 * np.pi * 10 * t) +
        0.5 * np.sin(2 * np.pi * 20 * t) +
        0.2 * np.sin(2 * np.pi * 40 * t)
    )
    
    # Add random white noise
    synthetic_eeg += np.random.normal(0, 0.5, len(t))
    
    print(f"Extracting features from synthetic EEG data ({duration}s at {fs}Hz)...\n")
    features = extract_band_power(synthetic_eeg, fs)
    
    for band, power in features.items():
        print(f"{band:5s} Band Power: {power:.4f}")
