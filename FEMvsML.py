import json
import pandas as pd
import glob

# -----------------------------
# USER SETTINGS
# -----------------------------
FEM_SOLVE_TIME = 120.0  # seconds per FEM simulation

# -----------------------------
# Collect metric files
# -----------------------------
metric_files = glob.glob("*/metrics_results_*_*.json")

rows = []

for file in metric_files:
    with open(file, "r") as f:
        data = json.load(f)

    model = data.get("model", "MIFNO")  # fallback
    split = data["split"]

    rows.append({
        "Model": model,
        "Split": split,
        "L2 Temp (%)": data["l2_temp_percent"],
        "L2 Stress (%)": data["l2_stress_percent"],
        "Inference Time (s/sample)": data["inference_time_per_sample_s"],
        "Speedup vs FEM":
            FEM_SOLVE_TIME / data["inference_time_per_sample_s"]
    })

df = pd.DataFrame(rows)

# sort nicely
df = df.sort_values(["Model", "Split"])

print("\n📊 MODEL COMPARISON")
print(df)

# save
df.to_csv("model_comparison.csv", index=False)
df.to_latex("model_comparison.tex", index=False)

print("\n✅ Saved:")
print("model_comparison.csv")
print("model_comparison.tex")
