
import torch
import torch.nn as nn
from tensordict.nn.distributions import NormalParamExtractor
from tensordict.nn import TensorDictModule, TensorDictSequential
from tensordict import TensorDict
from torchrl.collectors import SyncDataCollector
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator, SafeModule, ActorValueOperator
from torchrl.envs import Compose, ObservationNorm, StepCounter, TransformedEnv, Transform, ParallelEnv
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data import Unbounded
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchrl.envs.utils import set_exploration_type, ExplorationType

from models.pendulum_wrapper import TorchRLPendulumEnv
from models.torchrl_ac import ActorCritic

import multiprocessing
import math


class FrameLimitedTrainer():
    def __init__(self, device, env, actor_critic: ActorCritic, frame_limit=200,
                 total_frames_per_env=100_000, frames_per_batch_per_env=1000, minibatch_size=-1, num_epochs=10,
                 lr=1e-3, gamma=0.9, gae_lambda=0.9, clip_eps=0.2, entropy_eps=1e-4, critic_coeff=1.0,):
        
        self.device = device
        self.env = env
        self.actor_critic = actor_critic

        # Detect number of enviornments
        if isinstance(env, ParallelEnv):
            self.num_envs = env.batch_size[0]
        else:
            self.num_envs = 1

        self.total_frames = total_frames_per_env * self.num_envs
        self.frames_per_batch = frames_per_batch_per_env * self.num_envs
        self.num_epochs = num_epochs
        self.episodes_per_batch = frames_per_batch_per_env / frame_limit
        if minibatch_size == -1:
            self.minibatch_size = self.frames_per_batch//10
        else:
            self.minibatch_size = minibatch_size

        self.advantage_module = GAE(gamma=gamma, lmbda=gae_lambda, value_network=actor_critic.get_value_operator(), average_gae=True)
        self.loss_module = ClipPPOLoss(
            actor_network=actor_critic.get_policy_operator(),
            critic_network=actor_critic.get_value_operator(),
            clip_epsilon=clip_eps,
            entropy_bonus=bool(entropy_eps),
            entropy_coef=entropy_eps,
            critic_coef=critic_coeff,
            loss_critic_type="smooth_l1"
        )
        self.optimizer = Adam(self.loss_module.parameters(), lr=lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, self.total_frames // self.frames_per_batch)

        self.collector = SyncDataCollector(
            env,
            actor_critic.get_policy_operator(),
            frames_per_batch=self.frames_per_batch,
            total_frames=self.total_frames,
            split_trajs=False,
            device=device
        )

        self.replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=self.frames_per_batch, device="auto"),
            sampler=SamplerWithoutReplacement()
        )

    @torch.no_grad()
    @set_exploration_type(ExplorationType.DETERMINISTIC)
    def save(self, file_name="model.pt2"):
        policy = TensorDictSequential(self.actor_critic.get_policy_operator(), selected_out_keys=["action"]).requires_grad_(False).eval()

        if isinstance(self.env, ParallelEnv):
            observation = env.fake_tensordict()["observation"][0]
        else:
            observation = env.fake_tensordict()["observation"]

        model_export = torch.export.export(policy, args=(), kwargs={"observation": observation})

        torch.export.save(model_export, f"./saved_models/{file_name}")
        
        policy.requires_grad_(True).train()

    @torch.no_grad()
    def compute_avg_episode_returns(self, tensordict_data: TensorDict):
        rewards = tensordict_data["next"]["reward"].squeeze(-1)  # [num_envs, total_frames]
        dones = tensordict_data["next"]["done"].squeeze(-1)      # [num_envs, total_frames]
        
        all_episode_returns = []
        current_returns = torch.zeros(self.num_envs, device=self.device)
        
        for t in range(dones.shape[-1]):
            reward = rewards[:, t]
            done = dones[:, t]
            current_returns += reward
            if done.any():
                all_episode_returns.append(current_returns)
                current_returns = torch.zeros(self.num_envs, device=self.device)
        
        if all_episode_returns:
            return sum(all_episode_returns).mean() / len(all_episode_returns)
        else:
            return None


    def train(self):
        best_avg_episode_return = -torch.inf
        self.actor_critic.train()

        for i, tensordict_data in enumerate(self.collector):
            for _ in range(self.num_epochs):
                # Calculate Advantage, modifies tensordict_data inplace
                self.advantage_module(tensordict_data)
                data_view = tensordict_data.reshape(-1)
                #self.replay_buffer.extend(data_view.cpu())
                self.replay_buffer.extend(data_view)

                for _ in range(self.frames_per_batch // self.minibatch_size):
                    subdata = self.replay_buffer.sample(self.minibatch_size)
                    
                    #loss_vals = self.loss_module(subdata.to(device))
                    loss_vals = self.loss_module(subdata)
                    loss_value = (
                        loss_vals["loss_objective"]
                        + loss_vals["loss_critic"]
                        + loss_vals["loss_entropy"]
                    )

                    if torch.isnan(loss_vals["loss_objective"]) or torch.isnan(loss_vals["loss_critic"]) or torch.isnan(loss_vals["loss_entropy"]):
                        print("NAN LOSS DETECTED")

                    self.optimizer.zero_grad()
                    loss_value.backward()
                    torch.nn.utils.clip_grad_norm_(self.loss_module.parameters(), 1.0)
                    self.optimizer.step()

            self.scheduler.step()

            avg_episode_returns = self.compute_avg_episode_returns(tensordict_data)
            if avg_episode_returns > best_avg_episode_return:
                best_avg_episode_return = avg_episode_returns
                self.save()

            if avg_episode_returns:
                print(f"Step {i*self.frames_per_batch}, Episode {i*self.episodes_per_batch}, Average Return: {avg_episode_returns:.2f}, critic_loss {loss_vals["loss_critic"]}")
            else:
                print(f"Batch {i}, No completed episodes")



if __name__ == "__main__":
    is_fork = multiprocessing.get_start_method() == "fork"
    device = torch.device(0) if torch.cuda.is_available() and not is_fork else torch.device("cpu")

    num_envs = 10
    env = ParallelEnv(num_envs, TorchRLPendulumEnv, device=device)

    in_features = env.observation_spec["observation"].shape[-1]
    num_actions = env.action_spec.shape[-1]
    low = env.action_spec_unbatched.space.low
    high = env.action_spec_unbatched.space.high

    ac = ActorCritic(in_features, num_actions, low, high, 256).to(device)

    trainer = FrameLimitedTrainer(device, env, ac, total_frames_per_env=100000, frames_per_batch_per_env=1000, clip_eps=0.2)

    trainer.train()