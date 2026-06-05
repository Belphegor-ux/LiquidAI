"""Measure CPU inference latency (proxy for edge deployment, e.g. Raspberry Pi 5).

Run:  python benchmark.py
"""

import time

import numpy as np
import torch

import config
from evaluate import find_checkpoint
from model import CfCSeizurePredictor

WARMUP_RUNS = 3
TIMED_RUNS = 10


def main():
    model = CfCSeizurePredictor.load_from_checkpoint(find_checkpoint(), map_location="cpu")
    model.cpu().eval()

    # One 30-min window: (batch=1, time, channels).
    segment = torch.randn(1, config.SEQ_LEN, config.N_CHANNELS)

    with torch.no_grad():
        for _ in range(WARMUP_RUNS):
            model(segment)

        times = []
        for _ in range(TIMED_RUNS):
            start = time.perf_counter()
            model(segment)
            times.append(time.perf_counter() - start)

    avg_sec = float(np.mean(times))
    per_step_ms = avg_sec / config.SEQ_LEN * 1e3
    print(f"CPU inference: {avg_sec * 1e3:.1f} ms per 30-min window ({config.SEQ_LEN} steps)")
    print(f"Per-step latency: {per_step_ms:.3f} ms")


if __name__ == "__main__":
    main()
