# ROADMAP_IMPROVEMENTS.md

Hey Claude! We are upgrading the NeuroFlow architecture to match the 2023-2024 State-of-the-Art in EEG-based seizure prediction. I am going to start implementing the core architectural changes in `model.py` right now, but here is the comprehensive roadmap so we don't step on each other's toes.

## Objective
Boost our Sensitivity and lower the False Prediction Rate (FPR) by enhancing the `CfCSeizurePredictor` with spatial feature extraction, temporal attention, and cost-sensitive learning.

## Phase 1: Architectural Improvements (I am doing this now)
I am updating `model.py` to include:
1. **Spatial Embedding Layer:** A learned linear projection (acting as a 1x1 convolution over channels) to capture cross-channel spatial relationships *before* feeding the data into the CfC.
2. **Temporal Attention Pooling:** Changing the `CfC` to `return_sequences=True` to output all timesteps `(batch, time, hidden)`, and adding an Additive Attention layer to pool the sequence instead of just blindly taking the final hidden state. This allows the model to "pay attention" to anomalous preictal spikes anywhere in the 30-minute window.
3. **Cost-Sensitive Loss:** Replacing the standard `BCEWithLogitsLoss` with a `Focal Loss` implementation to heavily penalize the model for missing seizures (False Negatives), further addressing our extreme class imbalance.

## Phase 2: Feature Engineering (For Claude / Next Steps)
Once the architecture supports it, we need to upgrade our feature extraction in `preprocess.py`.
- **Current State:** Decimation of raw voltage.
- **Goal:** Please modify `preprocess.py` to extract **Time-Frequency features** (e.g., Short-Time Fourier Transform power bands: Alpha, Beta, Theta, Gamma) and/or **Sample Entropy** instead of just returning decimated raw voltage.
- *Note:* If you do this, make sure to update `config.N_CHANNELS` to reflect the new feature dimension size, and ensure it seamlessly feeds into the `(batch, time, features)` format expected by `model.py`.

## Phase 3: Graph Neural Networks (Future)
If the simple spatial embedding layer proves insufficient, we will map the 23 electrodes into a physical graph topology and replace the embedding layer with a standard GNN layer (like GraphSAGE or GCN). 
