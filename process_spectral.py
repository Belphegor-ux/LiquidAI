import numpy as np
import config
from extract_features import extract_band_power
import os
import sys

def main():
    print("Loading segments.npy...")
    segments = np.load(config.OUTPUT_DIR / "segments.npy") # (N, seq_len, channels)
    N, seq_len, channels = segments.shape
    
    # We want to extract features for each segment.
    # extract_band_power expects data of shape (channels, seq_len) for 2D.
    # Currently segments is (N, seq_len, channels), so we'll transpose to (N, channels, seq_len)
    
    print("Extracting spectral bands...")
    spectral_segments = []
    
    fs = config.EPOCH_SAMPLE_RATE # 128 Hz
    
    for i in range(N):
        if i % 100 == 0:
            print(f"Processing segment {i}/{N}...")
        
        data = segments[i].T # (channels, seq_len)
        band_powers = extract_band_power(data, fs) # returns dict of 5 bands, each shape (channels,)
        
        # Stack into (channels, 5 bands) or (5 bands, channels)
        # Let's stack as a vector of size (channels * 5)
        # Or (5, channels)
        bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
        stacked = np.stack([band_powers[b] for b in bands], axis=-1) # (channels, 5)
        
        # Flatten to 1D: size 23*5 = 115
        spectral_segments.append(stacked.flatten())
        
    spectral_segments = np.array(spectral_segments, dtype=np.float32)
    # Shape is (N, 115)
    # Since CfC expects (batch, time, features), we can shape it as (N, 1, 115)
    spectral_segments = spectral_segments.reshape(N, 1, channels * 5)
    
    out_path = config.OUTPUT_DIR / "segments_spectral.npy"
    print(f"Saving to {out_path} with shape {spectral_segments.shape}...")
    np.save(out_path, spectral_segments)
    print("Done!")

if __name__ == "__main__":
    main()
