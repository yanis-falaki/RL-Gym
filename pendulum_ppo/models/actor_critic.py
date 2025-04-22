import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

class PPOBackbone(nn.Module):
    def __init__(self, num_in_features, num_out_features=128):
        super(PPOBackbone, self).__init__()
        self.fc1 = nn.Linear(num_in_features, 128) 
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_out_features)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        return x
    

class PPOCriticHead(nn.Module):
    def __init__(self, in_features):
        super(PPOCriticHead, self).__init__()
        self.fc1 = nn.Linear(in_features, 1)
    
    def forward(self, x):
        return self.fc1(x)


class PPOActorHead(nn.Module):
    def __init__(self, in_features):
        super(PPOActorHead, self).__init__()
        self.fc_mean = nn.Linear(in_features, 1)
        self.fc_log_std = nn.Linear(in_features, 1)

    def forward(self, x):
        mean = self.fc_mean(x)
        std = F.softplus(self.fc_log_std(x))
        return mean, std
    

class PPOActorCritic(nn.Module):
    def __init__(self, in_features, bottleneck_size=128):
        super(PPOActorCritic, self).__init__()
        self.backbone = PPOBackbone(in_features, bottleneck_size)
        self.critic_head = PPOCriticHead(bottleneck_size)
        self.actor_head = PPOActorHead(bottleneck_size)
        self.log_8 = torch.log(torch.tensor(8)) # Used for logp_pi calculation

    def forward(self, x):
        x = self.backbone(x)
        state_value = self.critic_head(x)
        mean, std = self.actor_head(x)
        return mean, std, state_value
    
    def calculate_logp_pi_from_u(self, u, pi_u_distribution):
        # Computes logprob from gaussian then applies correction for 2*tanh().
        # 2*tanh() correction reformulated from log(2*(1-tanh^2(x))) to a more numerically stable version below
        # Avoids the e^x + e^-x which creates large numerically unstable numbers
        logp_pi = pi_u_distribution.log_prob(u) - (self.log_8 - 2*(u + F.softplus(-2*u)))
        return logp_pi

    def get_action_and_value(self, obs, deterministic=False):
        mean, std, state_value = self(obs)

        # Sample
        pi_u_distribution = dist.Normal(mean, std)
        if deterministic:
            u = mean
        else:
            u = pi_u_distribution.rsample() # Uses reparametrization trick
        
        logp_pi = self.calculate_logp_pi_from_u(u, pi_u_distribution)

        pi_action = 2*F.tanh(u)
        
        return pi_action, logp_pi, u, state_value
    
    def get_logp_pi_from_u_and_state_value(self, obs, u):
        mean, std, state_value = self(obs)
        pi_u_distribution = dist.Normal(mean, std)
        logp_pi = self.calculate_logp_pi_from_u(u, pi_u_distribution)

        return logp_pi, state_value