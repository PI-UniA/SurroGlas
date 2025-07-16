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
np.save("MIFNO/X_train_mifno.npy", X)
np.save("MIFNO/Y_train_mifno.npy", Y)

# === Normalize ===
x_scaler = StandardScaler()
X_scaled = x_scaler.fit_transform(X)

y_scaler = StandardScaler()
Y_reshaped = Y.reshape(Y.shape[0], -1)
Y_scaled = y_scaler.fit_transform(Y_reshaped).reshape(Y.shape)

np.save("MIFNO/x_scaler_mean_mifno.npy", x_scaler.mean_)
np.save("MIFNO/x_scaler_scale_mifno.npy", x_scaler.scale_)
np.save("MIFNO/y_scaler_mean_mifno.npy", y_scaler.mean_)
np.save("MIFNO/y_scaler_scale_mifno.npy", y_scaler.scale_)

# === Split ===
X_train, Y_train = X_scaled, Y_scaled
np.save("MIFNO/Y_train_mifno.npy", Y_train)
np.save("MIFNO/X_train_mifno.npy", X_train)


# === Define MIFNO-style CNN model ===
class SimpleMIFNO(nn.Module):
    def __init__(self, Nx, Nt):
        super().__init__()
        self.Nx, self.Nt = Nx, Nt
        self.fc1 = nn.Linear(4, 128) # inputs x hidden dimmension
        self.fc2 = nn.Linear(128, Nx * Nt) # maps spatiotemporal grid
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.out = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x).view(-1, 1, self.Nx, self.Nt)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.out(x).squeeze(1)

# === Training ===
Nx, Nt = 49, 200
model = SimpleMIFNO(Nx, Nt)
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
end_time = time.time()
print(f"⏱️ Training time: {end_time - start_time:.2f} seconds")

torch.save(model.state_dict(), "MIFNO/mifno_model.pt")


