"""CfC-100 seizure predictor (PyTorch Lightning).

Uses the ncps `CfC` module, which integrates the closed-form continuous-time
dynamics over the whole sequence internally -- far faster and more stable than
stepping through 460k timesteps in a Python loop.

Inputs are (batch, time, channels); the model emits a single preictal logit.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ncps.torch import CfC
from pytorch_lightning import LightningModule
from sklearn.metrics import roc_auc_score

import config


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        # targets: (batch,)
        # logits: (batch,)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class AdditiveAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, time, hidden_size)
        attn_weights = F.softmax(self.attention(x), dim=1)  # (batch, time, 1)
        context = torch.sum(attn_weights * x, dim=1)  # (batch, hidden_size)
        return context


class CfCSeizurePredictor(LightningModule):
    def __init__(
        self,
        input_size=config.N_CHANNELS,
        hidden_size=config.HIDDEN_SIZE,
        learning_rate=config.LEARNING_RATE,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Spatial Embedding Layer (1x1 conv over channels) to capture cross-channel relationships
        self.spatial_embed = nn.Sequential(
            nn.Linear(input_size, 64), nn.ReLU(), nn.Linear(64, input_size)
        )

        # CfC over the full sequence; return all sequences for temporal attention pooling
        self.cfc = CfC(input_size, hidden_size, batch_first=True, return_sequences=True)

        # Temporal Attention Pooling
        self.attention = AdditiveAttention(hidden_size)

        self.readout = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(64, 1),
        )
        # Cost-Sensitive Focal Loss (heavily penalize false negatives)
        self.criterion = FocalLoss(alpha=0.75, gamma=2.0)

        self._val_preds = []
        self._val_targets = []

    def forward(self, x):
        """x: (batch, time, channels) -> (batch,) logits."""
        # Spatial embedding applied at each timestep
        x = self.spatial_embed(x)  # (batch, time, input_size)

        out, _ = self.cfc(x)  # (batch, time, hidden_size)

        # Attention pooling over time
        context = self.attention(out)  # (batch, hidden_size)

        return self.readout(context).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, x):
        return torch.sigmoid(self(x))

    def training_step(self, batch, _batch_idx):
        x, y = batch
        loss = self.criterion(self(x), y.float())
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, _batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y.float())
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self._val_preds.append(torch.sigmoid(logits).detach().cpu())
        self._val_targets.append(y.detach().cpu())

    def on_validation_epoch_end(self):
        if not self._val_preds:
            return
        preds = torch.cat(self._val_preds).numpy()
        targets = torch.cat(self._val_targets).numpy()
        # AUROC is undefined with a single class present. Still log a neutral 0.5
        # so val_auc always exists -- otherwise the checkpoint/early-stop monitors
        # crash on degenerate val splits (preictal segments are scarce).
        if len(set(targets.tolist())) > 1:
            auc = roc_auc_score(targets, preds)
        else:
            auc = 0.5
        self.log("val_auc", auc, on_epoch=True, prog_bar=True)
        self._val_preds.clear()
        self._val_targets.clear()

    def configure_optimizers(self):
        # AdamW's decoupled weight decay regularizes to reduce cross-patient overfit.
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=config.WEIGHT_DECAY,
        )


if __name__ == "__main__":
    model = CfCSeizurePredictor()
    dummy = torch.randn(4, config.SEQ_LEN, config.N_CHANNELS)
    out = model(dummy)
    print(f"OK - output shape {tuple(out.shape)} (expected (4,))")
