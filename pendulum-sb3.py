import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Fixed callback that correctly tracks all episodes
class ImprovedEpisodeCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(ImprovedEpisodeCallback, self).__init__(verbose)
        self.total_episodes = 0
        self.running_reward = None
        self.last_len_episode_buffer = 0
        self.last_first_episode_info = None
        self.last_episode_count = 0

    def _on_step(self):
        episode_buffer_len = len(self.model.ep_info_buffer)
        if episode_buffer_len > self.last_len_episode_buffer:
            print(f"Episode: {episode_buffer_len} Reward: {self.model.ep_info_buffer[-1].get('r')}")
            self.last_len_episode_buffer = episode_buffer_len
            self.episode_count = self.last_len_episode_buffer
            self.last_first_episode_info = self.model.ep_info_buffer[-1]

        if len(self.model.ep_info_buffer) == 100 and self.model.ep_info_buffer[-1] != self.last_first_episode_info:
            print(f"Episode: {self.episode_count} Reward: {self.model.ep_info_buffer[-1].get('r')}")
            self.episode_count += 1
            self.last_first_episode_info = self.model.ep_info_buffer[-1]

        return True

# Create environment
def make_env():
    env = gym.make("Pendulum-v1")
    env = Monitor(env)
    return env

# Create vectorized environment
vec_env = make_vec_env(make_env, n_envs=1)

# Normalize the environment
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

# Define PPO model with custom network architecture
policy_kwargs = dict(
    net_arch=dict(
        pi=[128, 128, 128],  # Policy network
        vf=[128, 128, 128]   # Value function network
    )
)

# Create the PPO model
model = PPO(
    "MlpPolicy",
    vec_env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    verbose=0,
    device=device,
    policy_kwargs=policy_kwargs
)

# Create callback
callback = ImprovedEpisodeCallback()

# Train the model
print("Starting training...")
timesteps = 20000000
model.learn(total_timesteps=timesteps, callback=callback)

# Save the model
model_path = "ppo_pendulum"
model.save(model_path)
print(f"Model saved to {model_path}")

# Save normalized environment parameters
vec_env_path = "vec_normalize.pkl"
vec_env.save(vec_env_path)
print(f"Vectorized environment saved to {vec_env_path}")

# Test the trained model
print("Testing trained model...")
env = gym.make("Pendulum-v1", render_mode="human")
obs, _ = env.reset()

for i in range(1000):
    # Normalize observation based on saved statistics
    obs = vec_env.normalize_obs(obs)
    
    # Get action from the trained model
    action, _ = model.predict(obs, deterministic=True)
    
    # Execute action in the environment
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, _ = env.reset()

env.close()