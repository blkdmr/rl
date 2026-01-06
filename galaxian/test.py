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

def process_obs(obs: np.ndarray) -> torch.Tensor:
    base_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=(0.485,0.456,0.406), std=(0.229, 0.224,0.225))
    ])

    img = Image.fromarray(obs.astype(np.uint8))
    x = base_transform(img)
    x = x.to(dtype=torch.float32)
    return x.unsqueeze(0)

@app.entrypoint
def main(args):

    env = gym.make("ALE/Galaxian-v5", obs_type="rgb", render_mode="rgb_array")

    env = gym.wrappers.RecordVideo(
        env,
        video_folder=args["export"]["video_folder"]
    )

    n_actions = int(env.action_space.n)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------------------------------------
    model = timm.create_model('resnet18', pretrained=True)

    for param in model.parameters():
        param.requires_grad = args["model"]["backbone"]

    model.fc = nn.Linear(model.fc.in_features, n_actions)
    model.load_state_dict(torch.load(args["export"]["model"], weights_only=True))
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
