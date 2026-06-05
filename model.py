"""CfC-100 seizure predictor (PyTorch Lightning).

Uses the ncps `CfC` module, which integrates the closed-form continuous-time
dynamics over the whole sequence internally -- far faster and more stable than
stepping through 460k timesteps in a Python loop.

Inputs are (batch, time, channels); the model emits a single preictal logit.
"""

import torch
import torch.nn as nn
from ncps.torch import CfC
from pytorch_lightning import LightningModule
from sklearn.metrics import roc_auc_score

import config


class CfCSeizurePredictor(LightningModule):
    def __init__(self, input_size=config.N_CHANNELS, hidden_size=config.HIDDEN_SIZE,
                 learning_rate=config.LEARNING_RATE):
        super().__init__()
        self.save_hyperparameters()

        # CfC over the full sequence; last hidden state -> readout.
        self.cfc = CfC(input_size, hidden_size, batch_first=True, return_sequences=False)
        self.readout = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )
        # Logits + BCEWithLogits is numerically stable under 16-bit mixed precision
        # (plain Sigmoid + BCELoss can produce NaNs there).
        self.criterion = nn.BCEWithLogitsLoss()

        self._val_preds = []
        self._val_targets = []

    def forward(self, x):
        """x: (batch, time, channels) -> (batch,) logits."""
        out, _ = self.cfc(x)          # (batch, hidden_size)
        return self.readout(out).squeeze(-1)

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
        # AUROC is undefined with a single class present; guard for early epochs.
        if len(set(targets.tolist())) > 1:
            self.log("val_auc", roc_auc_score(targets, preds), on_epoch=True, prog_bar=True)
        self._val_preds.clear()
        self._val_targets.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


if __name__ == "__main__":
    model = CfCSeizurePredictor()
    dummy = torch.randn(4, config.SEQ_LEN, config.N_CHANNELS)
    out = model(dummy)
    print(f"OK - output shape {tuple(out.shape)} (expected (4,))")
