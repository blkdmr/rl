
import numpy as np
from dataclasses import dataclass
import typing as tt
import torch
import torch.nn as nn
import torch.optim as optim

import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE = 70
MAX_BATCHES = 10

class MLP(nn.Module):
    def __init__(self, obs_size: int, hidden_size: int, n_actions: int):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x: torch.Tensor):
        return self.model(x)

@dataclass
class EpisodeStep:
    observation: np.ndarray
    action: int

@dataclass
class Episode:
    reward: float
    steps: tt.List[EpisodeStep]

# Create a trigger function (records only if we have reached MAX_BATCHES)
# Note: Since RecordVideo tracks environment episodes, not your "batches",
# we can trigger it based on a global flag.
def last_batch_trigger(episode_id):
    # This will be updated inside your loop
    return getattr(model, 'record_now', False)

def record_episode(model, net):
    model.record_now = True
    obs, _ = model.reset()
    done = False
    sm = nn.Softmax(dim=1)

    print("Recording final evaluation episode...")
    while not done:
        # Match the logic used in generate_batches
        obs_v = torch.tensor(obs.flatten(), dtype=torch.float32).unsqueeze(0)
        act_probs_v = sm(net(obs_v))
        # Use Argmax for a 'cleaner' video of the learned policy
        action = np.argmax(act_probs_v.detach().numpy()[0])

        next_obs, reward, terminated, truncated, _ = model.step(action)
        done = terminated or truncated
        obs = next_obs.flatten()

    model.record_now = False
    print("Video saved successfully.")

def generate_batches(env: gym.Env,
                    model: MLP,
                    batch_size: int) -> tt.Generator[tt.List[Episode], None, None]:

    # the final batch of episodes
    batch = []

    # reset the env and get the first observation
    obs, _ = env.reset()

    episode_reward = 0.0
    episode_steps = []

    # used to extract a list of action probabilities
    # from the nn model
    sm = nn.Softmax(dim=1)

    while True:
        obs_v = torch.tensor(obs.flatten(), dtype=torch.float32)
        act_probs_v = sm(model(obs_v.unsqueeze(0))) # retrieve the action probabilities for the first observation
        act_probs = act_probs_v.detach().numpy()[0]

        action = np.random.choice(len(act_probs), p=act_probs) # choose an action using that distribution

        next_obs, reward, is_done, is_trunc, _ = env.step(action) # perfom the action

        episode_reward += float(reward)
        step = EpisodeStep(observation=obs.flatten(), action=action)
        episode_steps.append(step)

        if is_done or is_trunc:
            e = Episode(reward=episode_reward, steps=episode_steps)
            batch.append(e)

            # resets everything
            episode_reward = 0.0
            episode_steps = []
            next_obs, _ = env.reset()
            if len(batch) == batch_size:
                yield batch
                batch = []

        obs = next_obs.flatten()


def filter_batch(batch: tt.List[Episode], percentile: float) -> \
        tt.Tuple[torch.FloatTensor, torch.LongTensor, float, float]:
    rewards = list(map(lambda s: s.reward, batch))
    reward_bound = float(np.percentile(rewards, percentile))
    reward_mean = float(np.mean(rewards))

    train_obs: tt.List[np.ndarray] = []
    train_act: tt.List[int] = []
    for episode in batch:
        if episode.reward < reward_bound:
            continue
        train_obs.extend(map(lambda step: step.observation, episode.steps))
        train_act.extend(map(lambda step: step.action, episode.steps))

    train_obs_v = torch.FloatTensor(np.vstack(train_obs))
    print(f"TRAIN OBS V: {train_obs_v.shape}")
    train_act_v = torch.LongTensor(train_act)
    return train_obs_v, train_act_v, reward_bound, reward_mean


# Loading the enviroment
env = gym.make("CartPole-v1", render_mode="rgb_array")

env = gym.wrappers.RecordVideo(
    env,
    video_folder="video",
    episode_trigger=lambda x: getattr(env, 'record_now', False)
)

obs_size = 33600#env.observation_space.shape[0] #33600
n_actions = int(env.action_space.n)

print(obs_size)
print(n_actions)

# Defining the model
model = MLP(obs_size, HIDDEN_SIZE, n_actions)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=0.01)

batches = 0
env.record_now = False # Initialize the flag

for iter_no, batch in enumerate(generate_batches(env, model, BATCH_SIZE)):
    obs_v, acts_v, reward_b, reward_m = filter_batch(batch, PERCENTILE)
    optimizer.zero_grad()
    action_scores_v = model(obs_v)
    loss_v = loss_fn(action_scores_v, acts_v)
    loss_v.backward()
    optimizer.step()
    print("%d: loss=%.3f, reward_mean=%.1f, rw_bound=%.1f" % (
        iter_no, loss_v.item(), reward_m, reward_b))

    batches += 1

    if batches == MAX_BATCHES:
        record_episode(env, model)
        break

env.close()