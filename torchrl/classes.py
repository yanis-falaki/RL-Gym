import torch.nn as nn
from tensordict.nn.distributions import NormalParamExtractor
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator

class SharedBackbone(nn.Module):
    def __init__(self, in_features, num_cells):
        super(SharedBackbone, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, num_cells),
            nn.Tanh(),
            nn.Linear(num_cells, num_cells),
            nn.Tanh(),
            nn.Linear(num_cells, num_cells),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


class ActorNet(nn.Module):
    def __init__(self, backbone, num_actions):
        super(ActorNet, self).__init__()
        self.backbone = backbone
        num_cells = backbone.net[-2].out_features
        self.head = nn.Sequential(
            nn.Linear(num_cells, 2 * num_actions),
            NormalParamExtractor()
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


class ValueNet(nn.Module):
    def __init__(self, backbone):
        super(ValueNet, self).__init__()
        self.backbone = backbone
        num_cells = backbone.net[-2].out_features
        self.head = nn.Linear(num_cells, 1)

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)