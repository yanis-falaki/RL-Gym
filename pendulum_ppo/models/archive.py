import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

class PPOActor(nn.Module):
    def __init__(self, num_features):
        super(PPOActor, self).__init__()
        self.fc1 = nn.Linear(num_features, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc_mean = nn.Linear(128, 1)
        self.fc_log_std = nn.Linear(128, 1)

        self.log_8 = torch.log(torch.tensor(8)) # Used for logp_pi calculation
        
    def calculate_logp_pi_from_u(self, u, pi_u_distribution):
        # Computes logprob from gaussian then applies correction for 2*tanh().
        # 2*tanh() correction reformulated from log(2*(1-tanh^2(x))) to a more numerically stable version below
        # Avoids the e^x + e^-x which creates large numerically unstable numbers
        logp_pi = pi_u_distribution.log_prob(u) - (self.log_8 - 2*(u + F.softplus(-2*u)))
        return logp_pi
    
    def forward(self, x, ):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))

        mean = self.fc_mean(x)
        std = F.softplus(self.fc_log_std(x))

        return mean, std,

    def get_action(self, obs, deterministic=False):
        mean, std = self(obs)

        # Sample
        pi_u_distribution = dist.Normal(mean, std)
        if deterministic:
            u = mean
        else:
            u = pi_u_distribution.rsample() # uses reparameterization trick

        logp_pi = self.calculate_logp_pi_from_u(u, pi_u_distribution)

        pi_action = 2*F.tanh(u)

        return pi_action, logp_pi, u 
    
    def get_logp_pi_from_u(self, obs, u):
        mean, std = self(obs)
        pi_u_distribution = dist.Normal(mean, std)
        logp_pi = self.calculate_logp_pi_from_u(u, pi_u_distribution)
        return logp_pi

# Predicts state-value functions NOT action-value functions
class PPOCritic(nn.Module):
    def __init__(self, num_features):
        super(PPOCritic, self).__init__()
        self.fc1 = nn.Linear(num_features, 128) 
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 1) 
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x