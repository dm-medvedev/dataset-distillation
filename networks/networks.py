import torch.nn as nn
import torch.nn.functional as F

from . import utils


class LinearNet(utils.ReparamModule):
    def __init__(self, state):
        super(LinearNet, self).__init__()
        self.fc = nn.Linear(2, 1 if state.num_classes <= 2
                            else state.num_classes, bias=True)
        self.l2 = state.L2_coef

    def forward(self, x):
        out = self.fc(x)
        if self.training:
            for p in self.parameters():
                out = out + self.l2*(p**2).sum()
        return out


class NonLinearNet(utils.ReparamModule):
    def __init__(self, state, mid_sz=10):
        super(NonLinearNet, self).__init__()
        self.fc1 = nn.Linear(2, mid_sz)
        self.fc2 = nn.Linear(mid_sz, 1 if state.num_classes <= 2
                             else state.num_classes)

    def forward(self, x):
        out = F.relu(self.fc1(x), inplace=True)
        out = self.fc2(out)
        return out


class MoreNonLinearNet(utils.ReparamModule):
    def __init__(self, state, mid_sz=10):
        super(MoreNonLinearNet, self).__init__()
        self.fc1 = nn.Linear(2, mid_sz)
        self.fc2 = nn.Linear(mid_sz, mid_sz if state.num_classes <= 2
                             else state.num_classes)
        self.fc3 = nn.Linear(mid_sz, mid_sz if state.num_classes <= 2
                             else state.num_classes)
        self.fc4 = nn.Linear(mid_sz, 1 if state.num_classes <= 2
                             else state.num_classes)

    def forward(self, x):
        out = F.relu(self.fc1(x), inplace=True)
        out = F.relu(self.fc2(out), inplace=True)
        out = F.relu(self.fc3(out), inplace=True)
        out = self.fc4(out)
        return out
