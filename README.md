# SurroGlas – AI-driven Surrogate Modeling for Glass Annealing

## 📌 Introduction
SurroGlas is a research project focused on developing **AI-based surrogate models** for simulating thermo-mechanical processes in glass manufacturing, particularly in **annealing Lehr systems**.

The goal is to replace or accelerate traditional **Finite Element Method (FEM)** simulations using advanced **Neural Operator architectures**, enabling:

- ⚡ Real-time prediction of temperature and stress fields  
- 🔁 Rapid evaluation of process parameters  
- 🧠 Data-driven digital twin capabilities for glass production  

---

## 🧠 Core Concept

The repository combines:

- **FEM simulation (ground truth generation)**
- **AI surrogate models (MIFNO, MIONet)**
- **Interactive visualization (Streamlit dashboard)**

---

## 🏗️ Repository Structure (suggested)
SurroGlas/
│
├── fem/                     # FEM simulation (physics-based)
│   ├── ThermoViscoProblem.py
│   ├── ThermalModel.py
│   ├── ViscoelasticModel.py
│   └── geometry.py
│
├── training/                # Model training scripts
│   ├── train_MIFNO_multizone.py
│   └── train_MIONET.py
│
├── inference/               # Prediction scripts
│   ├── predict_MIFNO_multi-zone.py
│   └── predict_MIONET.py
│
├── apps/                    # User interface / visualization
│   └── streamlit_compare.py
│
├── utils/                   # Metrics and plotting utilities
│
├── data/                    # (ignored) generated datasets
├── results/                 # (ignored) simulation outputs
│
├── requirements.txt
├── environment.yml
└── README.md

---

## ⚙️ Models Implemented

### 🔹 FEM (Baseline)
- High-fidelity thermo-viscoelastic simulation
- Used for dataset generation and validation

### 🔹 MIFNO (Multi-Input Fourier Neural Operator)
- Learns full temperature & stress fields
- Handles varying process parameters:
  - Heat transfer coefficient (HTC)
  - Emissivity
  - Ambient temperature
  - Initial temperature

### 🔹 MIONet
- Alternative neural operator architecture
- Efficient multi-input mapping

---

## 📊 Key Features

- ✅ Multi-zone annealing Lehr simulation  
- ✅ Parameterized surrogate modeling  
- ✅ Temperature & stress prediction  
- ✅ KPI evaluation:
  - Relative L2 error  
  - Speed-up vs FEM  
  - Computation time  
- ✅ Interactive Streamlit dashboard  

---

## 🚀 Installation

### Local (recommended)

```bash
pip install -r requirements.txt
conda env create -f environment.yml
conda activate fenicsx-env

## ▶️ Usage

### 1. Run FEM simulation

```bash
python main.py

Run in parallel:
mpiexec -np N python main.py -parallel

### 2. Train AI models

python train_MIFNO_multizone.py
python train_MIONET.py

### 3. Run inference

python predict_MIFNO_multi-zone.py
python predict_MIONET.py

### 4. Launch Streamlit dashboard

streamlit run streamlit_compare.py


The dashboard allows:
	•	Comparison of FEM vs AI predictions
	•	Visualization of temperature and stress fields
	•	KPI analysis (error, speed-up, runtime)

## 📈 Example Outputs

The framework generates:
	•	Temperature field evolution over space and time
	•	Stress distribution in the glass
	•	Surface temperature profiles over the annealing lehr (cooling zones)
	•	Thickness-dependent temperature/stress maps
	•	Error maps between FEM and AI predictions
	•	Performance comparison plots

## 🧪 Research Context

This work contributes to:
  •	Numerical simulations and transient boundary conditions
	•	AI-based acceleration of numerical simulations
	•	Neural Operators for solving PDEs
	•	Data-driven multi-physics modeling
	•	Digital twins in glass manufacturing

The framework demonstrates how machine learning can approximate complex thermo-mechanical processes while significantly reducing computational cost.
