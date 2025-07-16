import numpy as np
import torch
import matplotlib.pyplot as plt
import time

# === Load data and scalers ===
X_train = np.load("MIONET/X_train_mionet.npy")
Y_train = np.load("MIONET/Y_train_mionet.npy")
x_mean = np.load("MIONET/x_scaler_mean_mionet.npy")
x_std = np.load("MIONET/x_scaler_scale_mionet.npy")
y_mean = np.load("MIONET/y_scaler_mean_mionet.npy")
y_std = np.load("MIONET/y_scaler_scale_mionet.npy")

# === MIONet model ===
class MIONet(torch.nn.Module):
    def __init__(self, input_dim, Nx, Nt):
        super().__init__()
        self.Nx, self.Nt = Nx, Nt
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU()
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(128, Nx * Nt),
        )
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 1, kernel_size=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x).view(-1, 1, self.Nx, self.Nt)
        x = self.conv(x)
        return x.squeeze(1)

Nx, Nt = 49, 200
model = MIONet(4, Nx, Nt)
model.load_state_dict(torch.load("MIONET/mionet_model.pt"))
model.eval()

# === Predict ===
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)

start = time.time()
Y_pred = model(X_train_tensor).detach().numpy()
inference_time = time.time() - start

# === Inverse transform ===
Y_pred_rescaled = Y_pred.reshape(len(Y_pred), -1) * y_std + y_mean
Y_test_rescaled = Y_train.reshape(len(Y_train), -1) * y_std + y_mean
Y_pred_rescaled = Y_pred_rescaled.reshape(-1, Nx, Nt)
Y_test_rescaled = Y_test_rescaled.reshape(-1, Nx, Nt)

# === Plot ===
plt.plot(Y_test_rescaled[0, :, 190], label='Y_true (FEM)', linestyle='--')
plt.plot(Y_pred_rescaled[0, :, 190], label='Y_pred (MIONet)')
plt.title("True vs Predicted Temperature Profiles over Space (case 0, t=190 seconds)")
plt.xlabel("Spatial Index")
plt.ylabel("Temperature (K)")
plt.legend()
plt.grid()
plt.show()

# === New input prediction ===
new_input = np.array([[100, 0.85, 273.1, 923.15]], dtype=np.float32)
new_input_scaled = (new_input - x_mean) / x_std
new_input_tensor = torch.tensor(new_input_scaled, dtype=torch.float32)

start_new = time.time()
new_pred_flat = model(new_input_tensor).detach().numpy()
inference_new = time.time() - start_new

new_pred = new_pred_flat.reshape(1, Nx, Nt)
new_pred_rescaled = new_pred.reshape(1, -1) * y_std + y_mean
new_pred_rescaled = new_pred_rescaled.reshape(1, Nx, Nt)

plt.plot(Y_test_rescaled[1, :, 190], label='True Temperature Case 6', linestyle='--')
plt.plot(new_pred_rescaled[0, :, 190], label='Predicted Temperature for new input')
plt.title("New Input Prediction at t=190")
plt.xlabel("Spatial Index")
plt.ylabel("Temperature (K)")
plt.legend()
plt.grid()
plt.show()

print(f"⏱️ Inference time (batch): {inference_time:.6f} seconds")
print(f"⏱️ Inference time (new input): {inference_new:.6f} seconds")

def relative_l2_norm(pred, ref):
    diff_norm = np.linalg.norm(pred - ref)
    ref_norm = np.linalg.norm(ref)
    return diff_norm / ref_norm

rel_l2 = [relative_l2_norm(Y_pred_rescaled[i], Y_test_rescaled[i]) for i in range(len(Y_pred_rescaled))]
print("📈 Mean relative L2 error:", np.mean(rel_l2))