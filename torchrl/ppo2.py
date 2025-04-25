
import torch
import torch.nn as nn
from tensordict.nn.distributions import NormalParamExtractor
from tensordict.nn import TensorDictModule
from tensordict import TensorDict
from torchrl.collectors import SyncDataCollector
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.envs import Compose, ObservationNorm, StepCounter, TransformedEnv
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from tutorial.pendulum_wrapper import TorchRLPendulumEnv
from classes import ActorNet, ValueNet, SharedBackbone

import multiprocessing
import math


class Trainer():
    def __init__(self, device, env, policy_module, value_module, total_frames=100_000, frames_per_batch=1000, minibatch_size=64, num_epochs=10,
                 lr=1e-3, gamma=0.9, gae_lambda=0.9, clip_eps=0.2, entropy_eps=1e-4, critic_coeff=1.0,):
        
        self.device = device
        self.env = env
        self.policy_module = policy_module
        self.value_module = value_module

        self.total_frames = total_frames
        self.frames_per_batch = frames_per_batch
        self.minibatch_size = minibatch_size
        self.num_epochs = num_epochs
        self.episodes_per_batch = frames_per_batch / 200

        self.advantage_module = GAE(gamma=gamma, lmbda=gae_lambda, value_network=value_module, average_gae=True)
        self.loss_module = ClipPPOLoss(
            actor_network=policy_module,
            critic_network=value_module,
            clip_epsilon=clip_eps,
            entropy_bonus=bool(entropy_eps),
            entropy_coef=entropy_eps,
            critic_coef=critic_coeff,
            loss_critic_type="smooth_l1"
        )
        self.optimizer = Adam(self.loss_module.parameters(), lr=lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, total_frames // frames_per_batch)

        self.collector = SyncDataCollector(
            env,
            policy_module,
            frames_per_batch=frames_per_batch,
            total_frames=total_frames,
            split_trajs=False,
            device=device
        )

        self.replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=frames_per_batch),
            sampler=SamplerWithoutReplacement()
        )

    @torch.no_grad()
    def compute_avg_episode_returns(self, tensordict_data: TensorDict):
            # Calculate average returns for completed episodes in the batch
            rewards = tensordict_data["next"]["reward"].squeeze(-1)  # Shape: [1000]
            dones = tensordict_data["next"]["done"].squeeze(-1)      # Shape: [1000]

            episode_returns = []
            current_return = 0
            for reward, done in zip(rewards, dones):
                current_return += reward.item()
                if done.item():
                    episode_returns.append(current_return)
                    current_return = 0
            
            if episode_returns:
                return sum(episode_returns) / len(episode_returns)   
            else:
                return None


    def train(self):
        for i, tensordict_data in enumerate(self.collector):
            for _ in range(self.num_epochs):
                # Calculate Advantage, modifies tensordict_data inplace
                self.advantage_module(tensordict_data)
                data_view = tensordict_data.reshape(-1)
                self.replay_buffer.extend(data_view.cpu())

                for _ in range(self.frames_per_batch // self.minibatch_size):
                    subdata = self.replay_buffer.sample(self.minibatch_size)

                    loss_vals = self.loss_module(subdata.to(device))
                    loss_value = (
                        loss_vals["loss_objective"]
                        + loss_vals["loss_critic"]
                        + loss_vals["loss_entropy"]
                    )

                    self.optimizer.zero_grad()
                    loss_value.backward()
                    torch.nn.utils.clip_grad_norm_(self.loss_module.parameters(), 1.0)
                    self.optimizer.step()

            avg_episode_returns = self.compute_avg_episode_returns(tensordict_data)
            if math.isnan(avg_episode_returns):
                print("NAN!")
            if avg_episode_returns:
                print(f"Step {i*self.frames_per_batch}, Episode {i*self.episodes_per_batch}, Average Return: {avg_episode_returns:.2f}, critic_loss {loss_vals["loss_critic"]}")
            else:
                print(f"Batch {i}, No completed episodes")
            
        self.scheduler.step()


if __name__ == "__main__":
    is_fork = multiprocessing.get_start_method() == "fork"
    device = torch.device(0) if torch.cuda.is_available() and not is_fork else torch.device("cpu")

    base_env = TorchRLPendulumEnv(device=device)
    env = TransformedEnv(
        base_env,
        Compose(
            StepCounter(),
        ),
    )

    in_features = env.observation_spec["observation"].shape[-1]

    in_features = env.observation_spec["observation"].shape[-1]

    # Create the shared backbone and networks
    backbone = SharedBackbone(in_features, 256).to(device)
    actor_net = ActorNet(backbone, env.action_spec.shape[-1]).to(device)
    value_net = ValueNet(backbone).to(device)

    policy_module = TensorDictModule(actor_net, in_keys=["observation"], out_keys=["loc", "scale"])
    policy_module = ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec,
        in_keys=["loc", "scale"],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": env.action_spec_unbatched.space.low,
            "high": env.action_spec_unbatched.space.high
        },
        return_log_prob=True,
    )

    value_module = ValueOperator(value_net, in_keys=["observation"])

    trainer = Trainer(device, env, policy_module, value_module, total_frames=1_000_000, minibatch_size=200)

    trainer.train()