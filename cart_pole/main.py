import gymnasium as gym
import numpy as np
import math
from fenn import Fenn

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


def log_kv(key: str, value):
    # Fenn prefixes timestamps; keep payload consistent with your example logs.
    print(f"{key}: {value}", flush=True)


class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = np.array(state, dtype=np.float64)
        self.parent = parent
        self.action = action
        self.children = []
        self.wins = 0.0
        self.visits = 0
        self.untried_actions = [0, 1]

    def select_child(self):
        log_total_visits = math.log(self.visits)
        return max(
            self.children,
            key=lambda c: (c.wins / c.visits) + 1.41 * math.sqrt(log_total_visits / c.visits),
        )

    def expand(self, sim_env):
        action = self.untried_actions.pop()
        obs, reward, terminated, truncated, _ = sim_env.step(action)
        child_node = MCTSNode(obs, parent=self, action=action)
        self.children.append(child_node)
        return child_node, terminated, truncated


def rollout(sim_env, cap_return=30.0):
    total_reward = 0.0
    terminated = False
    truncated = False
    while not (terminated or truncated) and total_reward < cap_return:
        _, reward, terminated, truncated, _ = sim_env.step(sim_env.action_space.sample())
        total_reward += float(reward)
    return total_reward


def path_actions(node):
    actions = []
    while node.parent is not None:
        actions.append(node.action)
        node = node.parent
    actions.reverse()
    return actions


def sync_env_to_node(sim_env, root_state, node):
    sim_env.reset()
    sim_env.unwrapped.state = np.array(root_state, dtype=np.float64)

    terminated = False
    truncated = False
    for a in path_actions(node):
        _, _, terminated, truncated, _ = sim_env.step(a)
        if terminated or truncated:
            break
    return terminated, truncated


def mcts_search(current_obs, iterations=50, rollout_cap=30.0):
    sim_env = gym.make("CartPole-v1", render_mode="rgb_array")
    sim_env = gym.wrappers.TimeLimit(sim_env, max_episode_steps=2000)

    root = MCTSNode(current_obs)

    for _ in range(iterations):
        node = root

        terminated, truncated = sync_env_to_node(sim_env, root.state, node)

        # 1) Selection
        while (not terminated) and (not truncated) and (not node.untried_actions) and node.children:
            node = node.select_child()
            terminated, truncated = sync_env_to_node(sim_env, root.state, node)

        # 2) Expansion
        if (not terminated) and (not truncated) and node.untried_actions:
            node, terminated, truncated = node.expand(sim_env)

        # 3) Simulation
        reward = 0.0 if (terminated or truncated) else rollout(sim_env, cap_return=rollout_cap)

        # 4) Backprop
        while node is not None:
            node.visits += 1
            node.wins += reward
            node = node.parent

    sim_env.close()

    if not root.children:
        return 0, 0, 0.0, 0.0

    best = max(root.children, key=lambda c: c.visits)

    root_mean = root.wins / max(1, root.visits)
    best_q = best.wins / max(1, best.visits)
    return best.action, root.visits, root_mean, best_q


@app.entrypoint
def main(args):
    # ---- print config like your fenn logs ----
    log_kv("project", args.get("project", "cartpole_mcts"))
    log_kv("logger/dir", args.get("logger", {}).get("dir", "logger"))

    env_max_steps = int(args.get("env", {}).get("max_episode_steps", 500))
    mcts_iters = int(args.get("mcts", {}).get("iterations", 40))
    rollout_cap = float(args.get("mcts", {}).get("rollout_cap", 30.0))

    log_kv("env/max_episode_steps", env_max_steps)
    log_kv("mcts/iterations", mcts_iters)
    log_kv("mcts/rollout_cap", rollout_cap)
    log_kv("export/video_folder", args.get("export", {}).get("video_folder", "video"))

    # ---- env ----
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = gym.wrappers.TimeLimit(env, max_episode_steps=env_max_steps)
    env = gym.wrappers.RecordVideo(env, video_folder=args.get("export", {}).get("video_folder", "video"))

    obs, _ = env.reset()
    terminated = False
    truncated = False

    ep_return = 0.0
    ep_len = 0
    step = 0

    # ---- rollout with per-step logging ----
    while not (terminated or truncated):
        action, root_visits, root_mean, best_q = mcts_search(
            obs, iterations=mcts_iters, rollout_cap=rollout_cap
        )

        obs, reward, terminated, truncated, _ = env.step(action)

        ep_return += float(reward)
        ep_len += 1
        step += 1

        # Emulate the "[k]: ..." style lines; here k is the env step.
        print(
            f"[{step}]: reward={float(reward):.1f}, "
            f"ep_return={ep_return:.1f}, "
            f"root_visits={root_visits:d}, "
            f"root_mean={root_mean:.3f}, "
            f"best_q={best_q:.3f}",
            flush=True,
        )

    print(f"[episode_end]: ep_return={ep_return:.1f}, ep_len={ep_len:d}", flush=True)

    env.close()


if __name__ == "__main__":
    app.run()
