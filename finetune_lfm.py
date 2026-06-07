import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW

import config

# Mock dataset mapping EEG risk probabilities to clinical text explanations
MOCK_DATA = [
    (0.85, "HIGH RISK. High-frequency paroxysmal bursts detected indicate rapid synchronization of cortical networks, which could trigger a runaway excitation and cause an immediate major seizure. Administer rescue medication if prescribed."),
    (0.92, "HIGH RISK. Sustained sharp wave activity signifies a breakdown of inhibitory mechanisms, potentially cascading into a secondary generalized seizure. Prepare for potential onset and monitor the patient continuously."),
    (0.55, "ELEVATED RISK. Occasional spike-and-wave discharges suggest an unstable thalamocortical loop, which might progressively lower the seizure threshold. Monitor the patient closely for any physical signs."),
    (0.60, "ELEVATED RISK. Focal rhythmic slowing observed in Fp1-Fp2 indicates localized network instability, which could propagate and cause a clinical event. Increase observation frequency and keep rescue plan ready."),
    (0.10, "LOW RISK. Normal background rhythm shows balanced neural excitation and inhibition, meaning an event is highly unlikely. No immediate action needed. Continue standard care."),
    (0.20, "LOW RISK. Isolated slowing without epileptiform features likely reflects benign physiological state variations without increasing synchronization risk. Patient is stable. Continue routine monitoring.")
]

SYSTEM_PROMPT = (
    "You are a clinical decision-support assistant for an epilepsy seizure-"
    "prediction system. Given a model's preictal probability (the chance a "
    "seizure begins within the next 30 minutes), write a brief, calm, actionable "
    "alert for a caregiver. Crucially, emphasize deep causal reasoning: state the risk level, "
    "describe the underlying neurophysiological anomaly, explain *why* it matters "
    "and *what it causes* (e.g. 'because of X, it could cause Y'), and suggest one "
    "concrete next step. Do not invent vitals or diagnoses."
)


class RiskExplanationDataset(Dataset):
    def __init__(self, tokenizer, data, max_length=128):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prob, explanation = self.data[idx]
        
        user_msg = f"Preictal probability: {prob * 100:.1f}%. Write the caregiver alert."
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": explanation}
        ]
        
        try:
            text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        except Exception:
            text = f"System: {SYSTEM_PROMPT}\nUser: {user_msg}\nAssistant: {explanation}"
        
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        
        # Labels are the same as input_ids for causal language modeling
        labels = input_ids.clone()
        # Ignore padding tokens in the loss computation
        labels[attention_mask == 0] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model {config.LLM_MODEL} on {device}...")
    
    tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL)
    if tokenizer.pad_token is None:
        # Some causal LMs do not have a pad token set by default
        tokenizer.pad_token = tokenizer.eos_token
        
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(config.LLM_MODEL, torch_dtype=dtype)
    model.to(device)
    model.train()
    
    dataset = RiskExplanationDataset(tokenizer, MOCK_DATA)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    optimizer = AdamW(model.parameters(), lr=5e-5)
    
    epochs = 3
    print("Starting fine-tuning...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{epochs}] - Average Loss: {avg_loss:.4f}")
        
    print("Training complete!")
    print("To save the model, uncomment: model.save_pretrained('models/finetuned_lfm')")


if __name__ == "__main__":
    main()
