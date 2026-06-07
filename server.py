import os
import random
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import config
from evaluate import find_checkpoint
from model import CfCSeizurePredictor

app = FastAPI(title="NeuroFlow API")

# Add CORS middleware to allow requests from the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global cache for model and data to avoid reloading on every request
model_cache = None
device_cache = None
segments_cache = None

def get_model_and_data():
    global model_cache, device_cache, segments_cache
    
    if model_cache is None:
        device_cache = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            ckpt = find_checkpoint()
            model_cache = CfCSeizurePredictor.load_from_checkpoint(ckpt, map_location=device_cache)
            model_cache.to(device_cache).eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
            
    if segments_cache is None:
        try:
            # Use memory mapping for the 1.6GB segments file to save RAM and load instantly
            segments_path = config.OUTPUT_DIR / "segments.npy"
            segments_cache = np.load(segments_path, mmap_mode='r')
        except Exception as e:
            raise RuntimeError(f"Failed to load data segments: {e}")
            
    return model_cache, device_cache, segments_cache

@app.get("/api/predict")
def predict():
    try:
        model, device, segments = get_model_and_data()
        
        # Take a random segment from the preprocessed data
        idx = random.randint(0, len(segments) - 1)
        sample = segments[idx]  # shape: (SEQ_LEN, channels)
        
        # Prepare tensor and run prediction
        x = torch.from_numpy(np.array(sample[None])).float().to(device)
        with torch.no_grad():
            prob = float(model.predict_proba(x).cpu().item())
            
        # Generate simulated channel activity data for the frontend visualization
        # We simulate 4 channels with 100 points each using a random walk to resemble EEG drift
        simulated_channels = []
        for _ in range(4):
            walk = np.cumsum(np.random.randn(100) * 0.1).tolist()
            simulated_channels.append(walk)
            
        return {
            "status": "success",
            "segment_index": idx,
            "predict_proba": prob,
            "risk_level": "HIGH" if prob > 0.5 else "LOW",
            "simulated_channels": simulated_channels
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Run the server locally on port 8000
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
