# NeuroFlow Architectural Notes & Explanations
*Use this guide to confidently explain the technical decisions behind the project.*

## 1. Data Pipeline & Memory Management
**How it works:** Raw `.edf` files (which are massive) are processed into 10-second epochs at 128 Hz.
**Key Decision:** Why did we use `np.memmap` in `server.py` and `preprocess.py`?
**Explanation:** EEG datasets are enormous (our processed `segments.npy` is over 1.6GB). If we loaded this directly into RAM, standard machines or edge devices would crash with an Out-Of-Memory (OOM) error. Memory mapping (`mmap_mode='r'`) allows the operating system to leave the data on the hard drive and only stream the exact 10-second chunk we need into RAM exactly when we need it. This makes the FastAPI server boot instantly and use almost zero RAM.

## 2. The Machine Learning Stack
**How it works:** 
1. `extract_features.py` converts raw voltage into Spectral Bands (Alpha, Beta, Gamma, etc.).
2. A Spatial Embedding layer mixes the 23 channels.
3. The CfC (Continuous-time RNN) processes the temporal sequence.
4. Additive Attention pools the sequence into a final risk score.

**Key Decision:** Why did we add spectral features instead of just using raw epochs?
**Explanation:** Originally, we tried feeding raw 128 Hz voltage directly into the CfC. However, cross-patient generalization was roughly at chance (AUROC ~0.475). The raw time-domain signal contains too much patient-specific noise. By extracting the Power Spectral Density (PSD), we isolate the actual *energy* in different brain wave frequencies. These energy shifts are much more uniform across different human brains during a preictal state.

## 3. Patient-Specific Calibration (`finetune.py`)
**How it works:** We take the generalized base model and train it slightly on the first 20% of a specific patient's timeline before evaluating on the remaining 80%.
**Key Decision:** Why do we do this?
**Explanation:** Epilepsy is highly idiosyncratic. A "spike" for Patient A might just be their normal resting state, while for Patient B it's an immediate seizure warning. By calibrating the model to the patient's baseline, we drastically lower the False Alarm Rate (FAR) to our target of `< 0.3/h`. 

## 4. The Liquid LFM Explainer (`finetune_lfm.py`)
**How it works:** An LLM that translates a numerical risk score into clinical text.
**Key Decision:** Why train the LFM to use "Causal Reasoning"?
**Explanation:** Doctors suffer from alert fatigue. If an alarm just says "85% Risk," they might ignore it. By training the LFM to explicitly state the cause and effect (*"Focal rhythmic slowing in Fp1-Fp2 disrupts the frontal network, which could cause a secondary generalized seizure"*), we provide actionable, clinical justification that a doctor can instantly verify by looking at the specific channels mentioned.

## 5. The Frontend Bridge
**How it works:** A React/Vite frontend polling a Python FastAPI backend.
**Key Decision:** Why separate them instead of using Streamlit?
**Explanation:** While Streamlit (`demo.py`) was great for our initial V1 testing, it is synchronous and struggles with complex, multi-pane interactive dashboards. By upgrading to a React frontend and a FastAPI backend, we decoupled the heavy PyTorch inference from the UI thread. This allowed us to build the real-time 24-hour interactive graphs and the dynamic Ward Overview without any UI freezing.
