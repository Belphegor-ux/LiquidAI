"""Download an LFM2 model from Hugging Face via parallel byte-range curl.

HF's Xet CDN throttles per-connection (~0.25 MB/s here) and the hf_xet /
hf_transfer clients stall on this network, but parallel range requests hit
~20 MB/s aggregate. This fetches the small config/tokenizer files normally and
the big model.safetensors in N parallel chunks, into models/<name>/ for loading
with transformers from a local path.

Run:  python download_lfm.py
"""

import concurrent.futures
import os
import subprocess
import sys
from pathlib import Path

# Always the HF repo id -- config.LLM_MODEL becomes a LOCAL path once downloaded,
# which would build an invalid URL on re-run.
REPO = "LiquidAI/LFM2.5-1.2B-Instruct"
DEST = Path("models") / REPO.split("/")[-1].lower()
BASE = f"https://huggingface.co/{REPO}/resolve/main"
NULL_DEVICE = "NUL" if os.name == "nt" else "/dev/null"
SMALL_FILES = [
    "config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
]
BIG_FILE = "model.safetensors"
N_STREAMS = 8


def _curl_to(url: str, dest: Path, extra: list[str] | None = None) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["curl", "-fsSL", *(extra or []), "-o", str(dest), url]
    return subprocess.run(cmd).returncode == 0


def _resolve_signed(url: str) -> str:
    out = subprocess.run(
        ["curl", "-s", "-o", NULL_DEVICE, "-w", "%{url_effective}", "-L", url],
        capture_output=True,
        text=True,
    )
    return out.stdout.strip()


def _content_length(url: str) -> int:
    out = subprocess.run(["curl", "-sIL", url], capture_output=True, text=True)
    sizes = [
        int(line.split(":", 1)[1])
        for line in out.stdout.splitlines()
        if line.lower().startswith("content-length")
    ]
    if not sizes:
        raise RuntimeError(f"no content-length for {url}")
    return sizes[-1]


def _download_big(name: str) -> None:
    dest = DEST / name
    signed = _resolve_signed(f"{BASE}/{name}")
    size = _content_length(signed)
    chunk = size // N_STREAMS
    ranges = []
    for i in range(N_STREAMS):
        start = i * chunk
        end = size - 1 if i == N_STREAMS - 1 else (start + chunk - 1)
        ranges.append((i, start, end))

    DEST.mkdir(parents=True, exist_ok=True)
    parts = [DEST / f"{name}.part{i}" for i, _, _ in ranges]
    print(f"{name}: {size / 1e9:.2f} GB in {N_STREAMS} parallel streams")

    def fetch(job):
        i, start, end = job
        ok = _curl_to(signed, parts[i], extra=["-r", f"{start}-{end}"])
        if not ok:
            raise RuntimeError(f"stream {i} failed")
        return i

    with concurrent.futures.ThreadPoolExecutor(max_workers=N_STREAMS) as ex:
        list(ex.map(fetch, ranges))

    with open(dest, "wb") as out:
        for part in parts:
            with open(part, "rb") as fh:
                while block := fh.read(8 << 20):
                    out.write(block)
            part.unlink()

    got = dest.stat().st_size
    if got != size:
        raise RuntimeError(f"size mismatch: got {got}, expected {size}")
    print(f"{name}: OK ({got / 1e9:.2f} GB)")


def main() -> None:
    print(f"Downloading {REPO} -> {DEST}")
    for name in SMALL_FILES:
        ok = _curl_to(f"{BASE}/{name}", DEST / name)
        print(f"  {name}: {'ok' if ok else 'skip (not in repo)'}")
    _download_big(BIG_FILE)
    print(f"\nDone. Load with: AutoModelForCausalLM.from_pretrained(r'{DEST}')")
    print(f"Set NEUROFLOW_LLM_MODEL={DEST} (or it's the config default path).")


if __name__ == "__main__":
    sys.exit(main())
