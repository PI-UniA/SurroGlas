import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
# === Load parameter and temperature files ===
results_dir = "results"
param_files = sorted([f for f in os.listdir(results_dir) if f.startswith("params_case_")])
temp_files = sorted([f for f in os.listdir(results_dir) if f.startswith("temperature_all_case")])

X_params = []
Y_temperatures = []

for param_file, temp_file in zip(param_files, temp_files):
    params = np.loadtxt(os.path.join(results_dir, param_file), skiprows=1, delimiter=",")
    temps = np.loadtxt(os.path.join(results_dir, temp_file))

    if params.ndim == 0:
        params = np.expand_dims(params, axis=0)

    X_params.append(params)
    Y_temperatures.append(temps)

X = np.stack(X_params).astype(np.float32)  # shape: (16, 4)
Y = np.stack(Y_temperatures).astype(np.float32)  # shape: (16, 9800)

# === Reshape Y to (16, 49, 200) for CNNs (space, time) ===
Y = Y.reshape(16, 200, 49).transpose(0, 2, 1)  # → (16, 49, 200)

# Save clean version for other models
np.save("MIONET/X_train_mionet.npy", X)
np.save("MIONET/Y_train_mionet.npy", Y)

# === Normalize ===
x_scaler = StandardScaler()
X_scaled = x_scaler.fit_transform(X)

y_scaler = StandardScaler()
Y_flat = Y.reshape(Y.shape[0], -1)
Y_scaled = y_scaler.fit_transform(Y_flat).reshape(Y.shape)

np.save("MIONET/x_scaler_mean_mionet.npy", x_scaler.mean_)
np.save("MIONET/x_scaler_scale_mionet.npy", x_scaler.scale_)
np.save("MIONET/y_scaler_mean_mionet.npy", y_scaler.mean_)
np.save("MIONET/y_scaler_scale_mionet.npy", y_scaler.scale_)

# === Split ===
X_train, Y_train = X_scaled, Y_scaled
np.save("MIONET/X_train_mionet.npy", X_train)
np.save("MIONET/Y_train_mionet.npy", Y_train)

# === MIONet model ===
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

# === Train ===
Nx, Nt = 49, 200
model = MIONet(input_dim=4, Nx=Nx, Nt=Nt)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)

start_time = time.time()
for epoch in range(1000):
    model.train()
    optimizer.zero_grad()
    pred = model(X_train_tensor)
    loss = loss_fn(pred, Y_train_tensor)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}")
print(f"\\n⏱️ Total training time: {time.time() - start_time:.2f} seconds")

torch.save(model.state_dict(), "MIONET/mionet_model.pt")