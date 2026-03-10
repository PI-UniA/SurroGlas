import json
import matplotlib.pyplot as plt
import os
import numpy as np

# === Load all metrics ===
models = ["MIFNO", "FEM"]
metrics = {}

for model in models:
    filename = f"metrics_results_{model}.json"
    if not os.path.exists(filename):
        print(f"⚠️ Missing metrics file for {model}: {filename}")
        continue
    with open(filename, "r") as f:
        metrics[model] = json.load(f)

# === Helper function to safely extract values ===
def get_metric(m, key):
    value = metrics.get(m, {}).get(key, None)
    return value if value is not None else np.nan

# === Prepare data ===
inference_times = [get_metric(m, "inference_time") for m in models]
l2_temp_errors  = [get_metric(m, "l2_temp") for m in models]
l2_stress_errors = [get_metric(m, "l2_stress") for m in models]

# === Plot 1: Inference Time ===
plt.figure(figsize=(3.35, 3))      
plt.bar(models, inference_times, color='lightgreen')
#plt.title("Inference Time Comparison")
plt.ylabel("Time (s)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.ylim(bottom=0)
plt.tight_layout()

# Export as JPG at 400 DPI (or higher)
plt.savefig("Figure9.jpg", dpi=600, bbox_inches='tight')
plt.show()

# === Plot 2: L2 Temperature Error ===
'''plt.figure(figsize=(7, 5))
plt.bar(models, l2_temp_errors, color='salmon')
plt.title("L2 Relative Error – Temperature")
plt.ylabel("Relative Error")
plt.grid(True, linestyle='--', alpha=0.5)
plt.ylim(bottom=0)
plt.tight_layout()
plt.show()

# === Plot 3: L2 Stress Error ===
plt.figure(figsize=(7, 5))
plt.bar(models, l2_stress_errors, color='plum')
plt.title("L2 Relative Error – Stress")
plt.ylabel("Relative Error")
plt.grid(True, linestyle='--', alpha=0.5)
plt.ylim(bottom=0)
plt.tight_layout()
plt.show()'''
