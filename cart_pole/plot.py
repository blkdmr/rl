#!/usr/bin/env python3
# plot.py  (call: python plot.py logfile)
import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


STEP_RE = re.compile(
    r"\[\d+\]:\s+reward=([-0-9.]+),\s+ep_return=([-0-9.]+),\s+root_visits=(\d+),\s+root_mean=([-0-9.]+),\s+best_q=([-0-9.]+)"
)

EP_END_RE = re.compile(r"\[episode_end\]:\s+ep_return=([-0-9.]+),\s+ep_len=(\d+)")


def parse(text: str):
    rewards, ep_returns, root_visits, root_mean, best_q = [], [], [], [], []
    for m in STEP_RE.finditer(text):
        rewards.append(float(m.group(1)))
        ep_returns.append(float(m.group(2)))
        root_visits.append(int(m.group(3)))
        root_mean.append(float(m.group(4)))
        best_q.append(float(m.group(5)))

    if not rewards:
        raise ValueError("No step lines matched. Check the log format / regex.")

    steps = np.arange(1, len(rewards) + 1, dtype=int)
    return (
        steps,
        np.array(rewards, float),
        np.array(ep_returns, float),
        np.array(root_visits, float),
        np.array(root_mean, float),
        np.array(best_q, float),
    )


def main():
    ap = argparse.ArgumentParser(description="Plot MCTS CartPole rollout diagnostics from stdout log.")
    ap.add_argument("logfile", type=Path)
    args = ap.parse_args()

    text = args.logfile.read_text(encoding="utf-8", errors="replace")
    steps, rewards, ep_returns, root_visits, root_mean, best_q = parse(text)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    axes[0].plot(steps, ep_returns, marker="o", linewidth=1)
    axes[0].set_ylabel("Episode return (running)")
    axes[0].set_title("CartPole rollout performance")
    axes[0].grid(True)

    axes[1].plot(steps, root_mean, marker="o", linewidth=1)
    axes[1].plot(steps, best_q, marker="o", linewidth=1)
    axes[1].set_xlabel("Environment step")
    axes[1].set_ylabel("Estimated value")
    axes[1].set_title("MCTS diagnostics")
    axes[1].legend(["Root mean value", "Chosen child mean value"])
    axes[1].grid(True)

    fig.tight_layout()

    fig.savefig("plot.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
