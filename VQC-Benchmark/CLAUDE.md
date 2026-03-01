# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VQC (Variational Quantum Classifier) benchmark comparing quantum ML models (VQC, QSVC) with classical baselines (SVM, MLP, Random Forest) on small datasets (Iris, Wine, Breast Cancer). Includes 7 experiments, 10-seed statistical analysis with Wilcoxon tests, 9 publication-quality figures, and a Korean-language LaTeX paper (IEEE format).

## Commands

```bash
pip install -r requirements.txt

# Run all tests (8-phase pipeline, 47+ tests)
python run_tests_dispatch.py

# Run specific test phase (1-8)
python run_tests_dispatch.py 3

# Run pytest directly
pytest tests/test_phase3_vqc_training.py -v

# Run experiments (10 seeds each, hours total)
python run_experiments_final.py           # Experiments 1-4
python run_experiment5_noise_final.py     # Experiment 5: Noise
python run_experiment6_convergence.py     # Experiment 6: Convergence
python run_experiment7_qsvc.py            # Experiment 7: QSVC

# Generate all 9 figures from result CSVs
python generate_figures.py

# Verify all paper tables match CSVs (228 values across 7 tables)
python verify_tables.py

# Build paper PDF (requires XeLaTeX + kotex)
xelatex paper.tex && xelatex paper.tex
```

## Architecture

**Data flow:** Dataset → StandardScaler → PCA → MinMaxScaler[0,π] → Model → Metrics → CSV → Figures → Paper

**Source modules (`src/`):**
- `data_loader.py` — Preprocessing pipeline. **Critical: StandardScaler MUST come before PCA** for correct variance decomposition. Iris skips PCA (already 4 features); Wine/BC get PCA 4D reduction.
- `vqc_model.py` — Feature maps (angle=RY gates, ZZ=ZZFeatureMap reps=1 fixed), ansatze (RealAmplitudes, EfficientSU2), optimizers (COBYLA, SPSA). Uses `algorithm_globals.random_seed` for Qiskit-internal reproducibility.
- `classical_models.py` — SVM (RBF kernel), MLP (hidden 64,32, max 500 epochs), Random Forest (100 trees).
- `stats_utils.py` — Wilcoxon signed-rank test (n<20) or paired t-test (n>=20), α=0.05.

**Experiments (root level, 7 total):**
- `run_experiments_final.py` — Exp 1: 8 models × 3 datasets; Exp 2: data size (20-100%, Iris+BC); Exp 3: COBYLA vs SPSA; Exp 4: circuit depth reps 1-4
- `run_experiment5_noise_final.py` — Depolarizing noise (AerSimulator, 2q error = 10× 1q per IBM convention)
- `run_experiment6_convergence.py` — TrackingCOBYLA wrapper records per-evaluation loss curves
- `run_experiment7_qsvc.py` — FidelityQuantumKernel with precomputed kernel SVM

**All experiments use 10 seeds:** `[42, 123, 456, 0, 1, 7, 13, 21, 77, 99]` with dual seeding (`np.random.seed` + `algorithm_globals.random_seed`).

**Tests (`tests/`):** 8 phases matching pipeline — data loading, preprocessing, VQC, classical, QSVC, aggregation, figures, paper build. Custom natural-language reporter in `conftest.py`. `run_tests_dispatch.py` runs each phase in a subprocess for isolation.

**Figures (`generate_figures.py`):** 9 figures (PNG+PDF, 300 DPI). Experiments 2-6 use dual-dataset subplot layouts (Iris | Breast Cancer side-by-side).

**Verification (`verify_tables.py`):** 7 dedicated verifiers parse LaTeX multirow tables and cross-check against CSVs. Tolerance: 0.0015.

## Key Conventions

- Results: `results/*.csv` with `_std` suffix columns for standard deviations
- Paper: Korean language, IEEE format (`paper.tex`), 20 references, 7 pages
- Qiskit stack: qiskit>=1.0, qiskit-aer>=0.13, qiskit-machine-learning>=0.7
- VQC uses `StatevectorSampler` (ideal, exp 1-4/6-7) or `AerSimulator` (noisy, exp 5) — results not directly comparable between simulators
- Feature map reps=1 always fixed; only ansatz reps varies (isolates circuit depth effect)
- CSV columns: train_accuracy, test_accuracy, train_f1, test_f1, train_time (each with `_std` pair), plus experiment-specific keys (model, dataset, fraction, reps, optimizer, error_rate, maxiter)
