import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


LINE_RE = re.compile(
    r"\[\d{4}-\d{2}-\d{2} [\d:]+\]\s+\[(\d+)\]:\s+loss=([0-9.]+),\s+reward_mean=([-0-9.]+),\s+rw_bound=([-0-9.]+)"
)


def parse_log(text: str):
    """
    Keeps the last occurrence per epoch (your log prints multiple lines per epoch).
    Returns: epochs, loss, reward_mean, rw_bound (numpy arrays).
    """
    per_epoch = {}
    for m in LINE_RE.finditer(text):
        ep = int(m.group(1))
        per_epoch[ep] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))

    if not per_epoch:
        raise ValueError("No training lines matched. Check input log format.")

    epochs = np.array(sorted(per_epoch.keys()), dtype=int)
    loss = np.array([per_epoch[e][0] for e in epochs], dtype=float)
    rmean = np.array([per_epoch[e][1] for e in epochs], dtype=float)
    rbound = np.array([per_epoch[e][2] for e in epochs], dtype=float)
    return epochs, loss, rmean, rbound


def main():
    ap = argparse.ArgumentParser(description="Plot CEM training curves from a Fenn stdout log.")
    ap.add_argument("logfile", type=Path, help="Path to the training log file")
    args = ap.parse_args()

    text = args.logfile.read_text(encoding="utf-8", errors="replace")
    epochs, loss, rmean, rbound = parse_log(text)

    # Single figure with two panels
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    axes[0].plot(epochs, rmean, marker="o", linewidth=1)
    axes[0].plot(epochs, rbound, marker="o", linewidth=1)
    axes[0].set_ylabel("Return")
    axes[0].set_title("Training performance")
    axes[0].legend(["Mean episode return", "Elite reward bound"])
    axes[0].grid(True)

    axes[1].plot(epochs, loss, marker="o", linewidth=1)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Cross-entropy loss")
    axes[1].set_title("Training loss")
    axes[1].grid(True)

    fig.tight_layout()

    fig.savefig("plot.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
