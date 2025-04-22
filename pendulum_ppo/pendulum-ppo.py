import gymnasium as gym
import numpy as np
from itertools import count
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dist
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

from models.actor_critic import PPOActorCritic

class TransitionBuffer():
    def __init__(self):
        self.pre_squashed_actions = []
        self.log_probs = []
        self.states = []
        self.rewards = []
        self.dones = []

    def save_transitions(self, pre_squashed_actions, log_probs, states, rewards,):
        self.pre_squashed_actions.append(pre_squashed_actions)
        self.log_probs.append(log_probs)
        self.states.append(states)
        self.rewards.append(rewards)

    def delete_transitions(self):
        del self.rewards[:]
        del self.pre_squashed_actions[:]
        del self.log_probs[:]
        del self.states[:]


class Trainer():
    def __init__(self, device, actor_critic: PPOActorCritic, actor_critic_optimizer,
                 gamma=0.9, epsilon=0.2, gae_lambda=0.90 ,
                 num_update_iters=10, num_episodes=500, num_envs=10,
                 save_path = "saved_models/"):
        self.device = device

        self.ac = actor_critic
        self.ac_optimizer = actor_critic_optimizer

        self.gamma = gamma
        self.epsilon = epsilon
        self.gae_lambda = gae_lambda

        self.num_update_iters = num_update_iters
        self.num_episodes = num_episodes
        self.num_envs = num_envs

        self.save_path = save_path

        self.transition_buffer = TransitionBuffer()

    def save_model(self, name="ac_model.pth"):
        model_path = os.path.join(self.save_path, name)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(self.ac.state_dict(), model_path)

    @torch.no_grad()
    def compute_monte_carlo_return(self, rewards: torch.Tensor):
        returns = torch.empty_like(rewards)
        returns[-1] = rewards[-1]

        for i in range(rewards.shape[0]-2, -1, -1):
            returns[i] = rewards[i] + self.gamma * returns[i+1]
            
        return returns

    @torch.no_grad()
    def compute_gae(self, rewards: torch.Tensor, approximated_values: torch.Tensor, normalize=True):
        advantages = torch.empty_like(rewards)
        
        next_advantage = 0
        next_value = 0
        
        for t in reversed(range(len(rewards))):
            # TD error
            delta = rewards[t] + self.gamma * next_value - approximated_values[t]
            
            # GAE formula
            advantages[t] = delta + self.gamma * self.gae_lambda * next_advantage
            
            # Update for next iteration
            next_advantage = advantages[t]
            next_value = approximated_values[t]
        
        td_returns = advantages + approximated_values
        
        # Normalize advantages if requested
        if normalize:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, td_returns

    def finish_episode(self):
        pre_squashed_actions = torch.stack(self.transition_buffer.pre_squashed_actions)
        old_log_probs = torch.stack(self.transition_buffer.log_probs)
        states = torch.stack(self.transition_buffer.states)
        rewards = torch.stack(self.transition_buffer.rewards)

        for epoch in range(self.num_update_iters):
            new_log_probs, state_values = self.ac.get_logp_pi_from_u_and_state_value(states, pre_squashed_actions)

            # Calculate advantages and returns
            advantages, returns = self.compute_gae(rewards, state_values, normalize=True)

            # Critic Loss
            critic_loss = F.smooth_l1_loss(state_values, returns)
            
            # Actor Loss
            ratios = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Total Loss
            total_loss = actor_loss + critic_loss

            # Update Actor-Critic
            self.ac_optimizer.zero_grad()
            total_loss.backward()
            self.ac_optimizer.step()

        self.transition_buffer.delete_transitions()

        return critic_loss, actor_loss, total_loss

    def train(self):
        env = gym.vector.AsyncVectorEnv([lambda: gym.make("Pendulum-v1") for _ in range(self.num_envs)])
        best_mean_reward = -np.inf
        self.ac.train()

        for i_episode in range(self.num_episodes):
            states, infos = env.reset()

            mean_episode_reward = 0
            for t in count():
                states = torch.from_numpy(states).float().to(self.device)
                t_feature = torch.full((states.shape[0], 1), t, device=self.device)
                states = torch.cat((states, t_feature), dim=1)

                with torch.no_grad():
                    action, log_prob, pre_squashed_action, state_values = self.ac.get_action_and_value(states)
                
                new_states, rewards, terminated, truncated, info = env.step(action.to('cpu').numpy())

                self.transition_buffer.save_transitions(pre_squashed_action.detach(),
                                                        log_prob.detach(),
                                                        states,
                                                        torch.from_numpy(rewards).to(self.device).unsqueeze(-1),)
                states = new_states
                mean_episode_reward += rewards.mean()
                if t >= 199:
                    break

            if mean_episode_reward > best_mean_reward:
                self.save_model()

            critic_loss, actor_loss, total_loss = self.finish_episode()
            print(f"Episode: {i_episode} Critic Loss: {critic_loss} Average Reward: {mean_episode_reward}")

        env.close()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ac = PPOActorCritic(in_features=4).to(device)
    ac_optimizer = optim.Adam(ac.parameters(), lr=1e-3)

    trainer = Trainer(device, ac, ac_optimizer, epsilon=0.2, num_episodes=2000, num_envs=10)
    trainer.train()