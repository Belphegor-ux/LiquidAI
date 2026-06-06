"""Streamlit demo: load a sample EEG window and show the preictal prediction.

Run:  streamlit run demo.py   (opens http://localhost:8501)
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch

import config
from evaluate import find_checkpoint
from model import CfCSeizurePredictor

st.set_page_config(page_title="NeuroFlow Seizure Predictor", layout="wide")
st.title("NeuroFlow: Cross-Patient Seizure Prediction")
st.markdown("Real-time preictal detection using **Liquid Neural Networks (CfC)** on **CHB-MIT**.")


@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CfCSeizurePredictor.load_from_checkpoint(find_checkpoint(), map_location=device)
    return model.to(device).eval(), device


@st.cache_data
def load_segments():
    return np.load(config.OUTPUT_DIR / "segments.npy")  # (N, SEQ_LEN, channels)


@st.cache_data
def load_patient_ids():
    path = config.OUTPUT_DIR / "patient_ids.json"
    return json.loads(path.read_text()) if path.exists() else None


model, device = load_model()
segments = load_segments()
patient_ids = load_patient_ids()

with st.sidebar:
    st.header("Model Info")
    st.metric("Architecture", f"CfC-{config.HIDDEN_SIZE}")
    st.metric(
        "Epoch", f"{config.EPOCH_SECONDS}s = {config.SEQ_LEN} steps @ {config.EPOCH_SAMPLE_RATE} Hz"
    )
    st.metric("Channels", config.N_CHANNELS)
    if st.button("Load random segment"):
        st.session_state.pop("idx", None)

if "idx" not in st.session_state:
    st.session_state["idx"] = int(np.random.randint(0, len(segments)))
idx = st.session_state["idx"]
sample = segments[idx]  # (SEQ_LEN, channels)
patient_id = patient_ids[idx] if patient_ids else None

col1, col2 = st.columns(2)
with col1:
    st.subheader("Input EEG")
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    for ch in range(4):
        axes[ch].plot(sample[:500, ch], linewidth=0.8)
        axes[ch].set_ylabel(f"Ch {ch + 1}")
        axes[ch].grid(True, alpha=0.3)
    axes[-1].set_xlabel("Timestep")
    fig.tight_layout()
    st.pyplot(fig)
    st.caption("First 4 channels of a 30-min window")

with col2:
    st.subheader("Prediction")
    x = torch.from_numpy(sample[None]).float().to(device)
    prob = float(model.predict_proba(x).cpu().item())

    c1, c2 = st.columns(2)
    c1.metric("Preictal probability", f"{prob * 100:.1f}%")
    (c2.warning if prob > 0.5 else c2.success)("HIGH RISK" if prob > 0.5 else "Low risk")

    fig2, ax = plt.subplots(figsize=(8, 2))
    ax.barh([0], [prob], color="red" if prob > 0.5 else "green")
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Preictal probability")
    ax.grid(True, axis="x", alpha=0.3)
    st.pyplot(fig2)
    if patient_id:
        st.caption(f"Segment {idx} - patient {patient_id}")

    st.subheader("Clinical alert (LFM2.5-1.2B)")
    if st.button("Generate alert"):
        with st.spinner("Liquid LFM2.5 generating clinical alert..."):
            try:
                from llm_explainer import explain_prediction

                st.session_state["alert"] = (idx, explain_prediction(prob, patient_id))
            except Exception as exc:  # noqa: BLE001 - keep the demo usable without the LFM
                st.session_state["alert"] = None
                st.error(f"LFM unavailable: {exc}")

    alert = st.session_state.get("alert")
    if alert and alert[0] == idx:  # only show the alert for the current segment
        st.info(alert[1])
    st.caption("On-device Liquid LFM2 explanation layer (run download_lfm.py first)")

st.markdown("---")
st.markdown(
    "**NeuroFlow** - edge-optimized seizure prediction proof-of-concept. "
    "CHB-MIT | CfC liquid network (prediction) + Liquid LFM2.5 (clinical alert) | "
    "cross-patient split, fully on-device."
)
