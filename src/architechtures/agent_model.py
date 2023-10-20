import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical



###################################################################
###### MPNet architechtures
###################################################################
class MPNet1(nn.Module):
    """Simple MLP policy mapping latent representation of obstacles to actions
    """
    def __init__(self, state_dim, action_dim):
        super(MPNet1, self).__init__()

        prob = 0.35
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 1024), nn.LeakyReLU(inplace=True),
            nn.Linear(1024, 1536), nn.LeakyReLU(inplace=True),
            nn.Dropout(p=prob),
            nn.Linear(1536, 1280), nn.LeakyReLU(inplace=True),
            nn.Dropout(p=prob),
            nn.Linear(1280, 1024), nn.LeakyReLU(inplace=True),
            nn.Dropout(p=prob),
            nn.Linear(1024, 768), nn.LeakyReLU(inplace=True), 
            nn.Linear(768, action_dim)
        )

    def forward(self, state):
        out = {}
        output = self.fc(state['state'])
        out['action'] = output # if training 
        # out = output
        return out