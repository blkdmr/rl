#!/usr/bin/env python3
# dqn_carracing_v3_processobs_tb_progress.py

import os
import time
import random
from dataclasses import dataclass
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Config:
    env_id: str = "CarRacing-v3"
    seed: int = 0

    total_steps: int = 1_000_000
    buffer_size: int = 200_000
    batch_size: int = 64
    gamma: float = 0.99
    lr: float = 1e-4

    learn_starts: int = 20_000
    train_every: int = 4
    target_update_every: int = 5_000

    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 200_000

    ckpt_path: str = "checkpoints/dqn_carracing_v3.pt"
    video_folder: str = "videos"

    tb_logdir: str = "runs/dqn_carracing_v3"
    tb_log_every: int = 1_000

    # PRINTS:
    print_every_steps: int = 5_000
    print_every_episodes: int = 1


cfg = Config()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def epsilon_by_step(step: int) -> float:
    return cfg.eps_end + (cfg.eps_start - cfg.eps_end) * np.exp(-step / cfg.eps_decay_steps)


def make_env(render_mode=None) -> gym.Env:
    # continuous=False => discrete actions (DQN-friendly) [web:5]
    return gym.make(cfg.env_id, continuous=False, render_mode=render_mode)


def process_obs(obs: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(obs).to(torch.float32) / 255.0
    x = x.permute(2, 0, 1).unsqueeze(0).contiguous()
    return x


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def add(self, s, a, r, s2, done):
        self.buf.append((s, a, r, s2, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = map(np.array, zip(*batch))
        return s, a, r, s2, d

    def __len__(self):
        return len(self.buf)


class DQN(nn.Module):
    def __init__(self, n_actions: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 96, 96)
            n_flat = self.features(dummy).shape[1]

        self.head = nn.Sequential(
            nn.Linear(n_flat, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        return self.head(self.features(x))


def save_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer, step: int):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "config": cfg.__dict__,
        },
        path,
    )


def load_checkpoint(path: str, model: nn.Module, device: str):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return ckpt


def train():
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_name = f"{time.strftime('%Y%m%d-%H%M%S')}_seed{cfg.seed}"
    writer = SummaryWriter(log_dir=os.path.join(cfg.tb_logdir, run_name))  # [web:93]

    env = make_env(render_mode=None)
    obs, _ = env.reset(seed=cfg.seed)

    n_actions = env.action_space.n
    q = DQN(n_actions).to(device)
    q_tgt = DQN(n_actions).to(device)
    q_tgt.load_state_dict(q.state_dict())

    opt = torch.optim.Adam(q.parameters(), lr=cfg.lr)
    rb = ReplayBuffer(cfg.buffer_size)

    ep_return, ep_len = 0.0, 0
    episode_idx = 0
    last_loss = None
    t0 = time.time()

    # PRINTS:
    print(f"[train] device={device} n_actions={n_actions} total_steps={cfg.total_steps}")
    print(f"[train] tensorboard_logdir={os.path.join(cfg.tb_logdir, run_name)}")

    for step in range(1, cfg.total_steps + 1):
        eps = epsilon_by_step(step)

        if random.random() < eps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                qvals = q(process_obs(obs).to(device))
                action = int(torch.argmax(qvals, dim=1).item())

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated  # Gymnasium episode end condition [web:10]

        rb.add(obs, action, reward, next_obs, done)
        obs = next_obs

        ep_return += reward
        ep_len += 1

        # episode end
        if done:
            episode_idx += 1
            writer.add_scalar("charts/episode_return", ep_return, step)  # [web:93]
            writer.add_scalar("charts/episode_length", ep_len, step)    # [web:93]

            # PRINTS:
            if episode_idx % cfg.print_every_episodes == 0:
                dt = time.time() - t0
                sps = step / max(dt, 1e-9)
                print(
                    f"[train] step={step}/{cfg.total_steps} "
                    f"episode={episode_idx} return={ep_return:.1f} len={ep_len} "
                    f"eps={eps:.3f} replay={len(rb)} sps={sps:.0f} "
                    f"loss={last_loss if last_loss is not None else 'NA'}"
                )

            obs, _ = env.reset()
            ep_return, ep_len = 0.0, 0

        # learn
        if len(rb) >= cfg.learn_starts and step % cfg.train_every == 0:
            s, a, r, s2, d = rb.sample(cfg.batch_size)

            s_t = torch.from_numpy(s).to(torch.float32) / 255.0
            s2_t = torch.from_numpy(s2).to(torch.float32) / 255.0
            s_t = s_t.permute(0, 3, 1, 2).contiguous().to(device)
            s2_t = s2_t.permute(0, 3, 1, 2).contiguous().to(device)

            a_t = torch.from_numpy(a).long().to(device)
            r_t = torch.from_numpy(r).float().to(device)
            d_t = torch.from_numpy(d.astype(np.float32)).float().to(device)

            q_sa = q(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                q_next = q_tgt(s2_t).max(dim=1).values
                target = r_t + cfg.gamma * (1.0 - d_t) * q_next

            loss = F.smooth_l1_loss(q_sa, target)
            last_loss = float(loss.detach().cpu().item())

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q.parameters(), 10.0)
            opt.step()

        if step % cfg.target_update_every == 0:
            q_tgt.load_state_dict(q.state_dict())

        # TensorBoard periodic
        if step % cfg.tb_log_every == 0:
            writer.add_scalar("charts/epsilon", eps, step)
            writer.add_scalar("charts/replay_size", len(rb), step)
            if last_loss is not None:
                writer.add_scalar("loss/td_huber", last_loss, step)

        # PRINTS: periodic “still running” line even if episodes are long
        if step % cfg.print_every_steps == 0:
            dt = time.time() - t0
            sps = step / max(dt, 1e-9)
            print(
                f"[train] step={step}/{cfg.total_steps} eps={eps:.3f} "
                f"replay={len(rb)} sps={sps:.0f} last_loss={last_loss}"
            )

        # checkpoint
        if step % 50_000 == 0:
            save_checkpoint(cfg.ckpt_path, q, opt, step)
            print(f"[train] saved checkpoint: {cfg.ckpt_path} (step {step})")

    save_checkpoint(cfg.ckpt_path, q, opt, cfg.total_steps)
    print(f"[train] done. final checkpoint saved: {cfg.ckpt_path}")

    env.close()
    writer.close()  # ensures event files are closed/flushed [web:93]


def test_model(model: nn.Module, device: str, video_folder: str):
    os.makedirs(video_folder, exist_ok=True)

    env: gym.Env = gym.make(
        cfg.env_id,
        render_mode="rgb_array",
        continuous=False,
    )

    env = gym.wrappers.RecordVideo(
        env,
        video_folder=video_folder,
        name_prefix="dqn-eval",
        episode_trigger=lambda ep: True,
    )

    obs, _ = env.reset()
    done = False

    print("[test] Recording final evaluation episode...")
    steps = 0
    total_reward = 0.0

    while not done:
        obs_v = process_obs(obs).to(device)
        with torch.no_grad():
            q_values = model(obs_v)
            action = int(torch.argmax(q_values, dim=1).item())

        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated  # [web:10]
        total_reward += float(reward)
        steps += 1

        if steps % 200 == 0:
            print(f"[test] step={steps} total_reward={total_reward:.1f}")

    print(f"[test] done. episode_steps={steps} total_reward={total_reward:.1f}")
    print("[test] Video saved successfully.")
    env.close()


def main():
    train()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tmp_env = make_env(render_mode=None)
    n_actions = tmp_env.action_space.n
    tmp_env.close()

    model = DQN(n_actions)
    load_checkpoint(cfg.ckpt_path, model, device)

    print(f"[main] loaded checkpoint from {cfg.ckpt_path}, starting video eval -> {cfg.video_folder}")
    test_model(model, device, cfg.video_folder)


if __name__ == "__main__":
    main()
