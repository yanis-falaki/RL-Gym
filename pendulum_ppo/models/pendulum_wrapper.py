import gymnasium as gym
import numpy as np
import torch
from typing import Optional, Union

from torchrl.data import Composite, Unbounded, Bounded
from tensordict import TensorDict
from torchrl.envs import EnvBase

# Gym Environment Wrapper
#   The one provided with torchrl doesn't work unless I downgrade my gym version due to 
#   breaking changes which were later fixed in gymnasium, nevertheless torchrl still 
#   doesn't support gymnasium version >= 1.0

class TorchRLPendulumEnv(EnvBase):
    """
    A TorchRL wrapper for the Gymnasium Pendulum-v1 environment.
    
    This wrapper converts the Gymnasium environment into a TorchRL compatible environment,
    handling the conversion between NumPy arrays and PyTorch tensors.
    """
    
    def __init__(
        self,
        device: Optional[Union[str, torch.device]] = None,
        batch_size: Optional[torch.Size] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(device=device, batch_size=batch_size)
        
        # Create the underlying Gymnasium environment
        self.env = gym.make("Pendulum-v1")
        
        if seed is not None:
            self.env.reset(seed=seed)
        
        # Define observation and action specs
        self.observation_spec = Composite(
            observation=Bounded(
                low=torch.tensor([-1.0, -1.0, -8.0, 0.0], device=self.device),
                high=torch.tensor([1.0, 1.0, 8.0, 1.0], device=self.device),
                shape=(4,),
                dtype=torch.float32,
                device=self.device,
            ),
            shape=(),
        )
        
        self.action_spec = Composite(
            action=Bounded(
                low=torch.tensor([-2.0], device=self.device),
                high=torch.tensor([2.0], device=self.device),
                shape=(1,),
                dtype=torch.float32,
                device=self.device,
            ),
            shape=(),
        )
        
        self.reward_spec = Composite(
            reward=Unbounded(shape=(1,), device=self.device),
            shape=(),
        )
        
        self.done_spec = Composite(
            done=Bounded(
                low=0,
                high=1,
                shape=(1,),
                dtype=torch.bool,
                device=self.device,
            ),
            terminated=Bounded(
                low=0,
                high=1,
                shape=(1,),
                dtype=torch.bool,
                device=self.device
            ),
            truncated=Bounded(
                low=0,
                high=1,
                shape=(1,),
                dtype=torch.bool,
                device=self.device
            ),
            shape=(),
        )

        self.frame = 0
    
    def _reset(
        self, 
        tensordict: Optional[TensorDict] = None,
        **kwargs
    ) -> TensorDict:
        """Reset the environment and return the initial state."""
        self.frame = 0

        if tensordict is not None and "seed" in tensordict:
            seed = tensordict.get("seed").item()
            kwargs["seed"] = seed
        
        obs, info = self.env.reset(**kwargs)
        obs = np.append(obs, self.frame)
        
        # Convert observation to torch tensor and create TensorDict
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
        
        tensordict = TensorDict({
            "observation": obs_tensor,
        }, batch_size=self.batch_size, device=self.device)
        
        return tensordict
    
    def _step(self, tensordict: TensorDict) -> TensorDict:
        """Take a step in the environment using the provided action."""
        self.frame += 1

        # Extract action from tensordict
        action = tensordict.get("action")
        
        # Convert action to numpy array
        action_np = action.cpu().detach().numpy().reshape(-1)
        
        # Step the environment
        next_obs, reward, terminated, truncated, info = self.env.step(action_np)
        next_obs = np.append(next_obs, self.frame/200)
        
        # Convert outputs to torch tensors
        next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
        reward_tensor = torch.tensor([reward], dtype=torch.float32, device=self.device)
        done_tensor = torch.tensor([terminated or truncated], dtype=torch.bool, device=self.device)
        terminated_tensor = torch.tensor([terminated], dtype=torch.bool, device=self.device)
        truncated_tensor = torch.tensor([truncated], dtype=torch.bool, device=self.device)

        self.last_reward_tensor = reward_tensor
        self.last_reward = reward
        self.last_action_np = action_np
        
        # Create output TensorDict
        out_tensordict = TensorDict({
            "observation": next_obs_tensor,
            "reward": reward_tensor,
            "terminated": terminated_tensor,
            "truncated": truncated_tensor
        }, batch_size=self.batch_size, device=self.device)
        
        return out_tensordict
    
    def _set_seed(self, seed: Optional[int] = None) -> None:
        """Set the seed for the environment."""
        if seed is not None:
            self.env.reset(seed=seed)
    
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render the environment."""
        try:
            return self.env.render()
        except Exception as e:
            print(f"Warning: render failed with error: {e}")
            return None
    
    def close(self) -> None:
        """Close the environment."""
        self.env.close()
        super().close()
    
    @property
    def state_spec(self) -> Composite:
        """Return the state specification."""
        return self.observation_spec