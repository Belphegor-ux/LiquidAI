"""Optional LFM layer: turn a CfC preictal probability into a clinical alert.

Runs a Liquid LFM2 instruct model in-process via transformers (on-device, no
server, no API key). Configured by the LLM_* values in config.py. This sits on
top of the CfC predictor -- it does NOT do seizure prediction itself.

Quick test:  python llm_explainer.py 0.82
The first call downloads the model (~2.4 GB) and caches it.
"""

import functools
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config

_SYSTEM_PROMPT = (
    "You are a clinical decision-support assistant for an epilepsy seizure-"
    "prediction system. Given a model's preictal probability (the chance a "
    "seizure begins within the next 30 minutes), write a brief, calm, actionable "
    "alert for a caregiver: state the risk level, what it means, and one concrete "
    "next step. Do not invent vitals or diagnoses. Two or three sentences."
)


def _resolve_device() -> str:
    if config.LLM_DEVICE == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return config.LLM_DEVICE


@functools.lru_cache(maxsize=1)
def _load():
    """Load tokenizer + model once (cached for the process)."""
    device = _resolve_device()
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL)
    model = AutoModelForCausalLM.from_pretrained(config.LLM_MODEL, dtype=dtype)
    model.to(device)
    model.eval()
    return tokenizer, model, device


def _risk_band(prob: float) -> str:
    if prob >= 0.80:
        return "HIGH"
    if prob >= 0.50:
        return "ELEVATED"
    return "LOW"


@torch.no_grad()
def explain_prediction(prob: float, patient_id: str | None = None) -> str:
    """Return a natural-language clinical alert for a preictal probability.

    Args:
        prob: preictal probability in [0, 1] from the CfC model.
        patient_id: optional identifier to include in the alert.

    Raises:
        ValueError: if prob is outside [0, 1].
    """
    if not 0.0 <= prob <= 1.0:
        raise ValueError(f"prob must be in [0, 1], got {prob}")

    tokenizer, model, device = _load()
    who = f" for patient {patient_id}" if patient_id else ""
    user_msg = (
        f"Preictal probability{who}: {prob * 100:.1f}% "
        f"(risk band: {_risk_band(prob)}). Write the caregiver alert."
    )

    inputs = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,  # include attention_mask for reliable generation
    ).to(device)

    output = model.generate(
        **inputs,
        do_sample=config.LLM_TEMPERATURE > 0,
        temperature=config.LLM_TEMPERATURE,
        max_new_tokens=config.LLM_MAX_TOKENS,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.eos_token_id,
    )
    # Decode only the newly generated tokens, not the prompt.
    new_tokens = output[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main() -> None:
    prob = float(sys.argv[1]) if len(sys.argv) > 1 else 0.82
    print(f"Model: {config.LLM_MODEL} on {_resolve_device()}")
    print(f"Preictal probability: {prob * 100:.1f}%  ({_risk_band(prob)})")
    print("-" * 60)
    print(explain_prediction(prob, patient_id="chb06"))


if __name__ == "__main__":
    main()
