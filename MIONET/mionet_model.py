# mionet_model.py
import torch
import torch.nn as nn

class MIONet(nn.Module):
    def __init__(self, input_dim, Nx, Nt):
        super().__init__()
        self.Nx, self.Nt = Nx, Nt
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, Nx * Nt),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x).view(-1, 1, self.Nx, self.Nt)
        x = self.conv(x)
        return x.squeeze(1)
