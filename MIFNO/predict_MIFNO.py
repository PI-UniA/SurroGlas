import numpy as np
import torch
import matplotlib.pyplot as plt
import time

# Load test set and scalers
#X_test = np.load("X_test.npy")
#Y_test = np.load("Y_test.npy")
X_train = np.load("MIFNO/X_train_mifno.npy")
Y_train = np.load("MIFNO/Y_train_mifno.npy")

x_mean = np.load("MIFNO/x_scaler_mean_mifno.npy")
x_std = np.load("MIFNO/x_scaler_scale_mifno.npy")
y_mean = np.load("MIFNO/y_scaler_mean_mifno.npy")
y_std = np.load("MIFNO/y_scaler_scale_mifno.npy")

# Model definition
class SimpleMIFNO(torch.nn.Module):
    def __init__(self, Nx, Nt):
        super().__init__()
        self.Nx, self.Nt = Nx, Nt
        self.fc1 = torch.nn.Linear(4, 128)
        self.fc2 = torch.nn.Linear(128, Nx * Nt)
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.out = torch.nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x).view(-1, 1, self.Nx, self.Nt)
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        return self.out(x).squeeze(1)

# Predict
Nx, Nt = 49, 200
model = SimpleMIFNO(Nx, Nt)
model.load_state_dict(torch.load("MIFNO/mifno_model.pt"))
model.eval()

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)

start = time.time()
Y_pred = model(X_train_tensor).detach().numpy()
inference_time = time.time() - start

# Inverse transform
Y_pred_rescaled = Y_pred.reshape(len(Y_pred), -1) * y_std + y_mean
Y_train_rescaled = Y_train.reshape(len(Y_train), -1) * y_std + y_mean
Y_pred_rescaled = Y_pred_rescaled.reshape(-1, Nx, Nt)
Y_train_rescaled = Y_train_rescaled.reshape(-1, Nx, Nt)


# === Plot prediction vs ground truth for test case 0 at t = 190 ===
plt.plot(Y_train_rescaled[0, :, 190], label='Y_temperatures (true/FEM)')
plt.plot(Y_pred_rescaled[0, :, 190], label='y_pred (MIFNO)')
plt.title("True vs Predicted Temperature Profiles over Space (test case 0, t=190)")
plt.xlabel("Spatial Index")
plt.ylabel("Temperature (K)")
plt.legend()
plt.grid()
plt.show()

# ------------------ Predict for New Input ------------------
new_input = np.array([[100, 0.85, 273.1, 923.15]], dtype=np.float32)
new_input_scaled = (new_input - x_mean) / x_std
new_input_tensor = torch.tensor(new_input_scaled, dtype=torch.float32)


start_new = time.time()
new_pred_flat = model(new_input_tensor).detach().numpy()
inference_new = time.time() - start_new

new_pred = new_pred_flat.reshape(1, Nx, Nt)
new_pred_rescaled = new_pred.reshape(1, -1) * y_std + y_mean
new_pred_rescaled = new_pred_rescaled.reshape(1, Nx, Nt)

# === Compare prediction at t = 190 with known test case 6 ===
plt.plot(Y_train_rescaled[1, :, 190], label='True Temperature Case 6', linestyle='--')
plt.plot(new_pred_rescaled[0, :, 190], label='Predicted Temperature for new input')
plt.title("New Input Prediction at t=190")
plt.xlabel("Spatial Index")
plt.ylabel("Temperature (K)")
plt.legend()
plt.grid()
plt.show()

# ------------------ Print Inference Times ------------------
print(f"⏱️  Inference time on all test data (batch of {len(X_train)}): {inference_time:.6f} seconds")
print(f"⏱️  Inference time for new input: {inference_new:.6f} seconds")

# ------------------ Relative L2 Error ------------------
def relative_l2_norm(pred, ref):
    diff_norm = np.linalg.norm(pred - ref)
    ref_norm = np.linalg.norm(ref)
    return diff_norm / ref_norm

rel_l2_set = [relative_l2_norm(Y_pred_rescaled[i], Y_train_rescaled[i]) for i in range(len(Y_pred_rescaled))]
print("📈 Mean relative L2 error:", np.mean(rel_l2_set))