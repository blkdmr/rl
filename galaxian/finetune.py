import numpy as np
from dataclasses import dataclass
import typing as tt
import torch
import torch.nn as nn
import torch.optim as optim
import timm
import torchvision.transforms as T
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

# ----------------------------
import gymnasium as gym
import ale_py
gym.register_envs(ale_py)
# ----------------------------

from fenn import Fenn
from fenn.notification import Notifier
from fenn.notification.services import Telegram

# ----------------------------
import warnings
warnings.filterwarnings(
    "ignore",
    message=r".*pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

warnings.filterwarnings(
    "ignore",
    message=r".*Overwriting existing videos at .*",
    category=UserWarning,
)
# ----------------------------

app = Fenn()
app.disable_disclaimer()

notifier = Notifier()
notifier.add_service(Telegram)

@dataclass
class EpisodeStep:
    observation: torch.Tensor
    action: int

@dataclass
class Episode:
    reward: float
    steps: tt.List[EpisodeStep]

class Model(nn.Module):
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

#class Model(nn.Module):
#    def __init__(self, n_actions: int, timm_model:str, pretrained: bool = True, head_dim: int = 64):
#        super().__init__()
#
#        self.backbone = timm.create_model(
#            timm_model,
#            pretrained=pretrained,
#            num_classes=0,
#            global_pool="avg",
#        )
#
#        # Freeze backbone params
#        self.backbone.requires_grad_(False)  # sets requires_grad for all params in the module
#        self.backbone.eval()
#
#        nf = self.backbone.num_features
#
#        self.head = nn.Sequential(
#            nn.Linear(nf, head_dim),
#            nn.ReLU(),
#            nn.Linear(head_dim, head_dim),
#            nn.ReLU(),
#            nn.Linear(head_dim, n_actions),
#        )
#
#    def forward(self, x: torch.Tensor) -> torch.Tensor:
#        # no_grad reduces memory since no autograd graph/activations are stored for the backbone
#        with torch.no_grad():
#            feats = self.backbone(x)
#        q = self.head(feats)
#        return q

def process_obs(obs: np.ndarray) -> torch.Tensor:
    base_transform = T.Compose([
        T.Resize((96, 96)),
        T.ToTensor(),
        T.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
    ])

    img = Image.fromarray(obs.astype(np.uint8))
    x = base_transform(img)
    x = x.to(dtype=torch.float32)
    return x.unsqueeze(0)

def generate_batch(env: gym.Env,
                    model: nn.Module,
                    device: str,
                    batch_size: int,
                    masked:tt.List[int]=None) -> tt.List[Episode]:
    model.eval()

    batch = []
    obs, _ = env.reset()

    episode_reward = 0.0
    episode_steps = []

    sm = nn.Softmax(dim=1)

    while len(batch) < batch_size:
        obs_v = process_obs(obs)
        obs_v = obs_v.to(device)
        with torch.no_grad():
            act_probs_v = sm(model(obs_v))

        act_probs = act_probs_v.detach().cpu().numpy()[0]

        if masked is not None:
            for command_id in masked:
                act_probs[command_id] = 0.0

            s = act_probs.sum()
            if s > 0:
                act_probs /= s
            else:
                act_probs = act_probs_v.detach().cpu().numpy()[0]

        action = np.random.choice(len(act_probs), p=act_probs)
        next_obs, reward, is_done, is_trunc, _ = env.step(action)

        episode_reward += float(reward)
        step = EpisodeStep(observation=obs_v.squeeze(0), action=action)
        episode_steps.append(step)

        if is_done or is_trunc:
            e = Episode(reward=episode_reward, steps=episode_steps)
            batch.append(e)

            episode_reward = 0.0
            episode_steps = []
            next_obs, _ = env.reset()

        obs = next_obs

    model.train()
    return batch


def filter_batch(batch, percentile):
    rewards = [ep.reward for ep in batch]
    reward_bound = float(np.percentile(rewards, percentile))
    reward_mean = float(np.mean(rewards))

    # determine the number of selected episodes ("elite")
    elite_count = sum(1 for ep in batch if ep.reward >= reward_bound)
    elite_frac = elite_count / max(1, len(batch))

    train_obs, train_act = [], []
    for episode in batch:
        if episode.reward < reward_bound:
            continue
        for step in episode.steps:
            train_obs.append(step.observation)
            train_act.append(step.action)

    obs_v = torch.stack(train_obs, dim=0)
    act_v = torch.as_tensor(train_act, dtype=torch.long)

    return obs_v, act_v, reward_bound, reward_mean, elite_frac


@app.entrypoint
def main(args):

    env = gym.make("ALE/Galaxian-v5", obs_type="rgb", render_mode="rgb_array")

    env = gym.wrappers.TimeLimit(env, max_episode_steps=args["env"]["max_episode_steps"])
    n_actions = int(env.action_space.n)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------------------------------------
    model = Model(n_actions)
    model.load_state_dict(torch.load("galaxian_100.pth"))
    model = model.to(device, dtype=torch.float32)
    # ---------------------------------------------------------
    
    # optimizer only sees trainable params
    optimizer = optim.Adam(
        params=(p for p in model.parameters() if p.requires_grad),
        lr=float(args["train"]["lr"]),
    )

    loss_fn = nn.CrossEntropyLoss()

    # TensorBoard writer
    logdir = args["export"]["tensorboard"]
    writer = SummaryWriter(log_dir=logdir)

    model.train()

    for epoch in range(1, args["sampling"]["epochs"] + 1):

        batch = generate_batch(env, model, device, args["sampling"]["batch_size"])

        # rollout stats from the raw batch
        ep_len_mean = float(np.mean([len(ep.steps) for ep in batch]))

        obs_v, acts_v, reward_b, reward_m, elite_frac = filter_batch(batch, args["sampling"]["percentile"])
        obs_v = obs_v.to(device)
        acts_v = acts_v.to(device)

        for i in range(20):
            optimizer.zero_grad()
            action_scores_v = model(obs_v)
            loss_v = loss_fn(action_scores_v, acts_v)
            loss_v.backward()

            optimizer.step()

            print(f"[{epoch}]: loss={loss_v.item():.6f}, reward_mean={reward_m:.1f}, rw_bound={reward_b:.1f}")

        # TensorBoard logging
        with torch.no_grad():
            probs = torch.softmax(action_scores_v, dim=1)
            entropy = (-(probs * torch.log(probs + 1e-8)).sum(dim=1)).mean().item()
            probs_mean = probs.mean(dim=0)  # (n_actions,)
            elite_acc = (action_scores_v.argmax(dim=1) == acts_v).float().mean().item()

        notifier.notify(f"Round {epoch} [ENDED] with loss {loss_v.item():.4f} and mean reward {reward_m:.1f}")

        writer.add_scalar("rollout/ep_rew_mean", reward_m, epoch)
        writer.add_scalar("rollout/ep_len_mean", ep_len_mean, epoch)
        writer.add_scalar("cem/reward_bound", reward_b, epoch)
        writer.add_scalar("cem/elite_frac", elite_frac, epoch)

        writer.add_scalar("train/loss_ce", loss_v.item(), epoch)
        writer.add_scalar("train/elite_acc", elite_acc, epoch)
        writer.add_scalar("policy/entropy", entropy, epoch)
        writer.add_scalar("optim/lr", optimizer.param_groups[0]["lr"], epoch)

        writer.add_scalars(
            "policy/action_prob_mean",
            {f"a{i}": probs_mean[i].item() for i in range(n_actions)},
            epoch
        )
        writer.add_scalar("train/elite_samples", int(obs_v.shape[0]), epoch)

    model.eval()
    model = model.to("cpu")
    torch.save(model.state_dict(), args["export"]["model"])

    env.close()
    writer.close()


if __name__ == "__main__":
    app.run()
