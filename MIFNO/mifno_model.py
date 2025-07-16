# mifno_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMIFNO(nn.Module):
    def __init__(self, Nx, Nt):
        super().__init__()
        self.Nx, self.Nt = Nx, Nt
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, Nx * Nt)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.out = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x).view(-1, 1, self.Nx, self.Nt)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.out(x).squeeze(1)
