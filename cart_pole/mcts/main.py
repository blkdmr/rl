import gymnasium as gym
import numpy as np
import math
import copy

class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_actions = [0, 1]

    def select_child(self):
        log_total_visits = math.log(self.visits)
        return max(self.children, key=lambda c: (c.wins / c.visits) + 1.41 * math.sqrt(log_total_visits / c.visits))

    def expand(self, sim_env):
        action = self.untried_actions.pop()
        # We don't deepcopy the WHOLE env here anymore, just set the state
        obs, reward, done, truncated, _ = sim_env.step(action)
        child_node = MCTSNode(obs, parent=self, action=action)
        self.children.append(child_node)
        return child_node

def rollout(sim_env):
    total_reward = 0
    done = False
    while not done and total_reward < 30:
        _, reward, done, truncated, _ = sim_env.step(sim_env.action_space.sample())
        total_reward += reward
    return total_reward

def mcts_search(current_obs, iterations=50):
    # Create a clean background env for simulation (no pygame/rendering)
    sim_env = gym.make("CartPole-v1") 
    
    root = MCTSNode(current_obs)
    
    for _ in range(iterations):
        node = root
        # Sync the sim_env to the current real-world state
        # In CartPole, the state is [cart_pos, cart_vel, pole_angle, pole_vel]
        sim_env.reset()
        sim_env.unwrapped.state = np.array(node.state)

        # 1. Selection
        while not node.untried_actions and node.children:
            node = node.select_child()
            sim_env.step(node.action)

        # 2. Expansion
        if node.untried_actions:
            node = node.expand(sim_env)

        # 3. Simulation
        reward = rollout(sim_env)

        # 4. Backpropagation
        while node is not None:
            node.visits += 1
            node.wins += reward
            node = node.parent
            
    sim_env.close()
    return max(root.children, key=lambda c: c.visits).action

# --- Main Loop ---
# This is the ONLY env that renders
real_env = gym.make("CartPole-v1", render_mode="human")
obs, info = real_env.reset()
done = False

while not done:
    # Pass the current observation to the search
    action = mcts_search(obs, iterations=40) 
    
    obs, reward, done, truncated, info = real_env.step(action)
    if truncated: done = True

real_env.close()