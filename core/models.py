import torch.nn as nn
import torch.nn.functional as F

class FCN(nn.Module):
    def __init__(self, state_dim, action_dim):
        """ single action for each state """
        super(FCN, self).__init__()
        self.affine1 = nn.Linear(state_dim, 32)
        self.affine2 = nn.Linear(32, 64)
        self.affine3 = nn.Linear(64, action_dim)

    def forward(self, x):
        """ get Q(s, a) for all a """
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))
        x = self.affine3(x)
        return x
