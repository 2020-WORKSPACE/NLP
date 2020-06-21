import torch
import torch.nn as nn
import torch.nn.functional as F

class DNN_model(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(5000, 1024)
        nn.init.kaiming_normal_(self.linear1.weight)
        self.relu1 = nn.ReLU()
        self.stage1 = nn.Sequential(self.linear1, self.relu1)

        self.linear2 = nn.Linear(1024, 256)
        nn.init.kaiming_normal_(self.linear2.weight)
        self.relu2 = nn.ReLU()
        self.stage2 = nn.Sequential(self.linear2, self.relu2)

        self.linear3 = nn.Linear(256, 256)
        nn.init.kaiming_normal_(self.linear3.weight)
        self.relu3 = nn.ReLU()
        self.stage3 = nn.Sequential(self.linear3, self.relu3)

        self.linear4 = nn.Linear(256, 1)
        nn.init.kaiming_normal_(self.linear3.weight)
        self.sigmoid4 = nn.Sigmoid()
        self.stage4 = nn.Sequential(self.linear4, self.sigmoid4)

        self.model = nn.Sequential(self.stage1, self.stage2, self.stage3, self.stage4)

    def forward(self, x):
        o = self.model(x.view(1, -1))
        return o