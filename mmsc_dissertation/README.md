# MMSC Dissertation Code

This directory contains the code used for the numerical experiments in the 2026 MMSC dissertation, Efficient Algorithms for Infinite-Dimensional Spectral Computations, by Margaret McCarthy.

## Requirements

All examples require the  complex version of Firedrake with SLEPc.

## Contents

- `dirac/` — Dirac operator examples
- `maxwell/` — Maxwell operator examples
- `advection_diffusion/` — advection--diffusion operator examples

## Running the code

The .py files that implement Algorithm 3 (alg3.py) can be run in parallel using MPI. For example, to run [`advection_alg3.py`](advection_diffusion/advection_alg3.py) with four MPI processes:

```bash 
mkdir -p advection_output

mkdir -p advection_output nohup env OUTDIR=advection_output mpiexec -n 4 python -u advection_alg3.py \ > advection_output/run.log 2>&1 < /dev/null &

