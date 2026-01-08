import numpy as np
import torch
import torch.nn as nn
import timm
import torchvision.transforms as T
from PIL import Image

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


@app.entrypoint
def main(args):

    env = gym.make("ALE/Galaxian-v5", obs_type="rgb", render_mode="rgb_array")

    #env = gym.wrappers.TimeLimit(env, max_episode_steps=args["env"]["max_episode_steps"])

    env = gym.wrappers.RecordVideo(
        env,
        video_folder=args["export"]["video_folder"]
    )

    n_actions = int(env.action_space.n)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------------------------------------
    model = Model(n_actions)
    model.load_state_dict(torch.load(args["export"]["model"]))
    model = model.to(device, dtype=torch.float32)
    # ---------------------------------------------------------

    obs, _ = env.reset()

    done = False
    sm = nn.Softmax(dim=1)

    print("Recording final evaluation episode...")
    while not done:
        obs_v = process_obs(obs)
        obs_v = obs_v.to(device)
        with torch.no_grad():
            act_probs_v = sm(model(obs_v))

        action = np.argmax(act_probs_v.detach().cpu().numpy()[0])

        next_obs, _ , terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        obs = next_obs

    print("Video saved successfully.")
    env.close()


if __name__ == "__main__":
    app.run()
