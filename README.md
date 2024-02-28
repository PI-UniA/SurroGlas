# SurroGlas

## Introduction

This is the working repository for the publicly funded research project "SurroGlas".
It contains the necessary files to run the Finite Element simulation that models float glass tempering.

## Structure

### `main.py`

At the top level, inputs are given in `main.py`.
From there, you can configure various aspects of the model, such as:

- the problem dimension using `problem_dim`,
- the solution method (Continuous or Discontinuous Galerkin) and order using `fe_config` and
- model parameters using the `model_params` dict.

### `ThermoViscoProblem`

This class holds the top-level data structure that includes the sub-models, as well as all data structures relevant for Finite Element analysis.

### `ThermalModel`

Here, all parameters are stored that are required to solve the heat equation.

### `ViscoelasticModel`

This class holds the necessary parameters, expressions and functions that are required to compute derived quantities in the viscoelastic material model, i.e. fictive temperature, shift function, scaled time and the various strain and stress increments.

## Installation

### Local

To install all required packages locally, a working Python installation is required.
In a corresponding environment, run within a shell:

```bash
pip3 install -r requirements.txt
```

### Docker

You might also want to use the pre-built docker container `dolfinx/dolfinx:v0.7.3` that the FEniCS project provides.
To execute the scripts in the container, clone the git repo and run the following command in a shell:

```bash
docker run -ti -v $(pwd):/root dolfinx/dolfinx:v0.7.3
```

This mounts the repository in the root folder of the container, in which a terminal session is opened.

## Running

From within the local Python environment where the required Python packages are installed, execute

```bash
python3 main.py
```
to run in serial.

For larger models, if you wish to simulate in parallel using `N` cores, execute

```bash
mpiexec -np N python3 main.py -parallel
````