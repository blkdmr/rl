import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

env = gym.make("ALE/DonkeyKong-v5", render_mode="rgb_array")
env = gym.wrappers.RecordVideo(
    env,
    video_folder="video"
)

obs, info = env.reset()
done = False
truncated = False

while not done and not truncated:
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)

env.close()