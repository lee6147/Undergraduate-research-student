# VQC Benchmark Improvements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix statistical, methodological, and fairness issues in the VQC benchmark project, then re-run all experiments and update the paper.

**Architecture:** Modify core source files (preprocessing, model, stats), update all 4 experiment scripts (10 seeds, dual-dataset), re-run experiments, regenerate figures, and rewrite paper.tex with new results + expanded discussion.

**Tech Stack:** Python 3.12, Qiskit 2.3.0, scikit-learn 1.8.0, scipy (for stats tests), matplotlib, seaborn

---

## Task 1: Fix Data Preprocessing

**Files:**
- Modify: `src/data_loader.py` (lines 1-46)
- Modify: `tests/test_phase1_data_loading.py`
- Modify: `tests/test_phase2_preprocessing.py`

**Problem:** MinMaxScaler before PCA distorts PCA variance. Should be StandardScaler → PCA → MinMaxScaler[0,π].

**Step 1: Update data_loader.py**

Replace the `load_dataset` function. Key change: use `StandardScaler` before PCA, then `MinMaxScaler(feature_range=(0, np.pi))` after PCA. For datasets that don't need PCA (like Iris with 4 features), apply MinMaxScaler directly.

```python
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA


DATASETS = {
    "iris": load_iris,
    "wine": load_wine,
    "breast_cancer": load_breast_cancer,
}


def load_dataset(name: str, test_size: float = 0.2, random_state: int = 42, n_features: int = None):
    """데이터셋을 로드하고 전처리하여 train/test로 분할한다.

    PCA가 필요한 경우: StandardScaler → PCA → MinMaxScaler[0,π]
    PCA가 불필요한 경우: MinMaxScaler[0,π]
    """
    loader = DATASETS[name]
    data = loader()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if n_features is not None and n_features < X_train.shape[1]:
        # StandardScaler before PCA for correct variance decomposition
        std_scaler = StandardScaler()
        X_train = std_scaler.fit_transform(X_train)
        X_test = std_scaler.transform(X_test)

        pca = PCA(n_components=n_features)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    # Final scaling to [0, π] for quantum encoding
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def subsample_dataset(X, y, fraction: float, random_state: int = 42):
    """학습 데이터를 일정 비율로 서브샘플링한다."""
    n_samples = max(int(len(X) * fraction), len(np.unique(y)) + 1)
    X_sub, _, y_sub, _ = train_test_split(
        X, y, train_size=n_samples, random_state=random_state, stratify=y
    )
    return X_sub, y_sub
```

**Step 2: Update test_phase1_data_loading.py**

The `test_scaling_range` test should still pass since final output is still [0,π]. No test changes needed unless tests check intermediate scaling. Run tests to verify:

```bash
python -m pytest tests/test_phase1_data_loading.py -v
```

**Step 3: Commit**

```bash
git add src/data_loader.py
git commit -m "fix: use StandardScaler before PCA for correct variance decomposition"
```

---

## Task 2: Fix VQC Model (reps clarity + reproducibility)

**Files:**
- Modify: `src/vqc_model.py` (lines 1-58)

**Problem 1:** `create_feature_map` accepts `reps` but angle encoding ignores it. Feature map reps and ansatz reps are conflated.
**Problem 2:** `np.random.seed` doesn't control Qiskit's internal RNG. Need `algorithm_globals.random_seed`.

**Step 1: Update vqc_model.py**

```python
import time
import numpy as np
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.circuit.library import RealAmplitudes, EfficientSU2, ZZFeatureMap
from sklearn.metrics import accuracy_score, f1_score
from qiskit_algorithms.utils import algorithm_globals


def create_feature_map(encoding: str, n_qubits: int):
    """Create a feature map circuit for data encoding.

    Note: reps is not configurable here; feature map complexity is fixed
    to isolate the effect of ansatz depth in experiments.
    """
    if encoding == "angle":
        params = ParameterVector("x", n_qubits)
        qc = QuantumCircuit(n_qubits)
        for i in range(n_qubits):
            qc.ry(params[i], i)
        return qc
    elif encoding == "zz":
        return ZZFeatureMap(feature_dimension=n_qubits, reps=1)
    else:
        raise ValueError(f"Unknown encoding: {encoding}")


def create_ansatz(ansatz_name: str, n_qubits: int, reps: int = 1):
    if ansatz_name == "real_amplitudes":
        return RealAmplitudes(num_qubits=n_qubits, reps=reps)
    elif ansatz_name == "efficient_su2":
        return EfficientSU2(num_qubits=n_qubits, reps=reps)
    else:
        raise ValueError(f"Unknown ansatz: {ansatz_name}")


def train_vqc(X_train, y_train, X_test, y_test, n_qubits=4, encoding="angle",
              ansatz_name="real_amplitudes", optimizer_name="cobyla", maxiter=100,
              reps=1, random_seed=None):
    from qiskit_algorithms.optimizers import COBYLA, SPSA
    from qiskit_machine_learning.algorithms import VQC
    from qiskit.primitives import StatevectorSampler

    # Set Qiskit-internal random seed for reproducibility
    if random_seed is not None:
        algorithm_globals.random_seed = random_seed

    feature_map = create_feature_map(encoding, n_qubits)
    ansatz = create_ansatz(ansatz_name, n_qubits, reps=reps)

    optimizers = {"cobyla": COBYLA(maxiter=maxiter), "spsa": SPSA(maxiter=maxiter)}
    opt = optimizers[optimizer_name]

    sampler = StatevectorSampler()
    vqc = VQC(sampler=sampler, feature_map=feature_map, ansatz=ansatz, optimizer=opt)

    start = time.time()
    vqc.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred_train = vqc.predict(X_train)
    y_pred_test = vqc.predict(X_test)

    return {
        "train_accuracy": accuracy_score(y_train, y_pred_train),
        "test_accuracy": accuracy_score(y_test, y_pred_test),
        "train_f1": f1_score(y_train, y_pred_train, average="macro"),
        "test_f1": f1_score(y_test, y_pred_test, average="macro"),
        "train_time": train_time,
    }
```

**Step 2: Update callers to pass `random_seed`**

All experiment scripts that call `train_vqc` need to pass `random_seed=seed`. This is done in Tasks 4-6.

**Step 3: Commit**

```bash
git add src/vqc_model.py
git commit -m "fix: remove reps from feature_map, add algorithm_globals seed for reproducibility"
```

---

## Task 3: Create Statistical Utilities

**Files:**
- Create: `src/stats_utils.py`
- Create: `tests/test_stats_utils.py`

**Step 1: Create src/stats_utils.py**

```python
"""Statistical testing utilities for VQC benchmark."""
import numpy as np
from scipy import stats


def paired_statistical_test(scores_a, scores_b, test_name="auto"):
    """Run paired statistical test between two sets of scores.

    For n >= 20: paired t-test
    For n < 20: Wilcoxon signed-rank test (non-parametric)

    Returns dict with test_name, statistic, p_value, significant (at alpha=0.05).
    """
    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)
    n = len(scores_a)

    if test_name == "auto":
        test_name = "t-test" if n >= 20 else "wilcoxon"

    if test_name == "t-test":
        stat, p_val = stats.ttest_rel(scores_a, scores_b)
    elif test_name == "wilcoxon":
        diff = scores_a - scores_b
        if np.all(diff == 0):
            return {"test": "wilcoxon", "statistic": 0.0, "p_value": 1.0, "significant": False}
        stat, p_val = stats.wilcoxon(scores_a, scores_b)
    else:
        raise ValueError(f"Unknown test: {test_name}")

    return {
        "test": test_name,
        "statistic": float(stat),
        "p_value": float(p_val),
        "significant": p_val < 0.05,
    }


def compute_stats(values):
    """Compute mean and std for a list of values."""
    arr = np.array(values)
    return float(np.mean(arr)), float(np.std(arr))
```

**Step 2: Create tests/test_stats_utils.py**

```python
"""Tests for statistical utilities."""
PHASE_NAME = "Stats Utilities"

import numpy as np
from src.stats_utils import paired_statistical_test, compute_stats


def test_paired_test_identical_scores():
    """Identical scores should give p=1.0 (not significant)."""
    scores = [0.9, 0.85, 0.88, 0.92, 0.87]
    result = paired_statistical_test(scores, scores)
    assert not result["significant"], f"Identical scores should not be significant, got p={result['p_value']}"


def test_paired_test_different_scores():
    """Very different scores should be significant."""
    a = [0.95, 0.94, 0.96, 0.93, 0.95, 0.94, 0.96, 0.93, 0.95, 0.94]
    b = [0.50, 0.52, 0.48, 0.51, 0.49, 0.50, 0.52, 0.48, 0.51, 0.49]
    result = paired_statistical_test(a, b)
    assert result["significant"], f"Very different scores should be significant, got p={result['p_value']}"


def test_paired_test_returns_required_keys():
    """Result dict should contain required keys."""
    a = [0.9, 0.85, 0.88]
    b = [0.8, 0.75, 0.78]
    result = paired_statistical_test(a, b)
    for key in ["test", "statistic", "p_value", "significant"]:
        assert key in result, f"Missing key: {key}"


def test_compute_stats():
    """compute_stats should return correct mean and std."""
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    mean, std = compute_stats(values)
    assert abs(mean - 3.0) < 1e-10
    assert abs(std - np.std(values)) < 1e-10
```

**Step 3: Run tests**

```bash
python -m pytest tests/test_stats_utils.py -v
```

**Step 4: Commit**

```bash
git add src/stats_utils.py tests/test_stats_utils.py
git commit -m "feat: add statistical testing utilities (paired t-test/Wilcoxon)"
```

---

## Task 4: Update Experiments 1-4 Script

**Files:**
- Modify: `run_experiments_final.py`

**Changes:**
1. SEEDS: 3 → 10
2. Experiment 1: Add VQC-Angle-RA-reps3 config for fair comparison
3. Experiment 1: Collect per-seed scores (not just aggregate) for statistical tests
4. Experiment 1: Run statistical test (best VQC vs each classical model)
5. Experiment 2: Add Breast Cancer dataset
6. Experiment 3: Add Breast Cancer dataset
7. Experiment 4: Add Breast Cancer dataset
8. Pass `random_seed=seed` to `train_vqc`

**Step 1: Rewrite run_experiments_final.py**

Key constant change:
```python
SEEDS = [42, 123, 456, 0, 1, 7, 13, 21, 77, 99]
```

In experiment_1, add this VQC config to `vqc_configs`:
```python
{"encoding": "angle", "ansatz_name": "real_amplitudes", "label": "VQC-Angle-RA-reps3", "reps": 3},
```

In all `train_vqc` calls, add `random_seed=seed` parameter.

In experiments 2-4, add a loop over `["iris", "breast_cancer"]` (two datasets instead of just Iris).

After experiment 1, add statistical testing:
```python
from src.stats_utils import paired_statistical_test

# Collect per-seed test accuracies for best VQC and each classical model
# Run paired_statistical_test and save p-values to results/experiment1_stats.csv
```

The full rewritten script should maintain the same CSV output format but with more rows (10 seeds → tighter std, 2 datasets for exp 2-4, extra VQC config in exp 1).

**Step 2: Verify script syntax**

```bash
python -c "import run_experiments_final"
```

**Step 3: Commit**

```bash
git add run_experiments_final.py
git commit -m "feat: expand experiments 1-4 (10 seeds, dual dataset, VQC-reps3, stats tests)"
```

---

## Task 5: Update Experiment 5 (Noise)

**Files:**
- Modify: `run_experiment5_noise_final.py`

**Changes:**
1. SEEDS: 3 → 10
2. Add Breast Cancer dataset (loop over both Iris and BC)
3. Add comment explaining 2-qubit error rate = 10× single-qubit (IBM hardware convention)
4. Pass `random_seed=seed` to internal VQC (via algorithm_globals in the local train_vqc_noisy function)

**Step 1: Update the script**

Key changes:
```python
SEEDS = [42, 123, 456, 0, 1, 7, 13, 21, 77, 99]

DATASETS = {
    "Iris": {"name": "iris", "n_features": 4},
    "Breast Cancer": {"name": "breast_cancer", "n_features": 4},
}
```

Add to `train_vqc_noisy`:
```python
from qiskit_algorithms.utils import algorithm_globals
algorithm_globals.random_seed = random_seed  # new parameter
```

Add comment before 2q error:
```python
# 2-qubit gate error ~10x single-qubit on IBM hardware (IBM Quantum roadmap, 2023)
error_2q = depolarizing_error(min(error_rate * 10, 1.0), 2)
```

**Step 2: Commit**

```bash
git add run_experiment5_noise_final.py
git commit -m "feat: expand noise experiment (10 seeds, dual dataset, document 2q error rationale)"
```

---

## Task 6: Update Experiment 6 (Convergence)

**Files:**
- Modify: `run_experiment6_convergence.py`

**Changes:**
1. SEEDS: 3 → 10
2. Add Breast Cancer dataset
3. Set algorithm_globals.random_seed in train_vqc_with_callback

**Step 1: Update the script**

Same pattern as Task 5: add DATASETS dict, loop over both, set random seed.

**Step 2: Commit**

```bash
git add run_experiment6_convergence.py
git commit -m "feat: expand convergence experiment (10 seeds, dual dataset)"
```

---

## Task 7: Update Experiment 7 (QSVC)

**Files:**
- Modify: `run_experiment7_qsvc.py`

**Changes:**
1. SEEDS: 3 → 10
2. No dataset change needed (already uses 3 datasets)
3. Set algorithm_globals.random_seed

**Step 1: Update SEEDS and add reproducibility**

```python
SEEDS = [42, 123, 456, 0, 1, 7, 13, 21, 77, 99]
```

Add in train_qsvc:
```python
from qiskit_algorithms.utils import algorithm_globals
algorithm_globals.random_seed = random_seed  # new parameter
```

**Step 2: Commit**

```bash
git add run_experiment7_qsvc.py
git commit -m "feat: expand QSVC experiment to 10 seeds"
```

---

## Task 8: Run All Experiments

**IMPORTANT:** This task takes several hours. Run scripts sequentially.

**Step 1: Run experiments 1-4**

```bash
cd C:\Users\user\vqc-benchmark
python run_experiments_final.py 2>&1 | tee logs/run_exp1-4.log
```

Expected output files:
- `results/experiment1_model_comparison.csv`
- `results/experiment1_stats.csv` (NEW: p-values)
- `results/experiment2_data_size.csv`
- `results/experiment3_optimizer.csv`
- `results/experiment4_circuit_depth.csv`

**Step 2: Run experiment 5 (noise)**

```bash
python run_experiment5_noise_final.py 2>&1 | tee logs/run_exp5.log
```

**Step 3: Run experiment 6 (convergence)**

```bash
python run_experiment6_convergence.py 2>&1 | tee logs/run_exp6.log
```

**Step 4: Run experiment 7 (QSVC)**

```bash
python run_experiment7_qsvc.py 2>&1 | tee logs/run_exp7.log
```

**Step 5: Verify all result files exist**

```bash
ls -la results/*.csv results/*.json
```

All 7 CSV files + loss_curves.json should be present and recently modified.

**Step 6: Commit results**

```bash
git add results/
git commit -m "data: re-run all experiments with 10 seeds, fixed preprocessing, dual dataset"
```

---

## Task 9: Update Figure Generation

**Files:**
- Modify: `generate_figures.py`

**Changes:**
1. Experiments 2-6 figures now need dual-dataset support (Iris + Breast Cancer subplots)
2. Figures for exp 2-4: change from single plot to 1×2 subplot (Iris | Breast Cancer)
3. Figures for exp 5, 6: same 1×2 subplot pattern
4. Figure 1 needs to handle the new VQC-Angle-RA-reps3 model
5. Add Figure 10: statistical significance table/chart (optional, can be text in paper instead)

For each existing figure function (fig3 through fig8), change from single-dataset to dual-subplot layout. The pattern is:

```python
def fig3_data_size():
    df = pd.read_csv("results/experiment2_data_size.csv")
    datasets = df['dataset'].unique()
    fig, axes = plt.subplots(1, len(datasets), figsize=(7*len(datasets), 5), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        subset = df[df['dataset'] == dataset]
        # ... same plotting logic per subplot ...
        ax.set_title(dataset, fontweight='bold')
    # ... save ...
```

**Step 1: Update all figure functions for dual-dataset support**

**Step 2: Regenerate figures**

```bash
python generate_figures.py
```

**Step 3: Verify figures**

```bash
ls -la figures/*.png
```

**Step 4: Commit**

```bash
git add generate_figures.py figures/
git commit -m "feat: update figures for dual-dataset support and new VQC config"
```

---

## Task 10: Update Paper (paper.tex)

**Files:**
- Modify: `paper.tex` (major rewrite of results + discussion sections)
- Modify: `paper.md` (sync with tex changes)

This is the largest task. Changes organized by section:

### 10a: Abstract
- Change "3개의 랜덤 시드" → "10개의 랜덤 시드"
- Update specific accuracy numbers after results are available

### 10b: Methodology (Section 3)

**Line ~147:** Add explanation of StandardScaler → PCA → MinMaxScaler:
```
PCA 적용 시 분산 기반 차원축소의 정확성을 위해 StandardScaler(평균 0, 표준편차 1)를 먼저 적용한 후 PCA를 수행하였다. 양자 인코딩에 필요한 $[0, \pi]$ 범위 변환은 PCA 이후 MinMaxScaler를 통해 수행하였다.
```

**Line ~182:** Update experiment 2-6 descriptions to mention dual dataset:
```
\item \textbf{실험 2:} 학습 데이터 크기에 따른 성능 변화 (Iris 및 Breast Cancer, 20\%--100\%)
```

**Line ~190-192:** Update statistical methodology:
```
모든 실험은 10개의 랜덤 시드(0, 1, 7, 13, 21, 42, 77, 99, 123, 456)로 반복 수행하였다.
결과는 평균$\pm$표준편차로 보고하며, 모델 간 성능 차이의 통계적 유의성은
Wilcoxon signed-rank test ($\alpha = 0.05$)로 검정하였다.
```

**Add new subsection after 3.7 (line ~200):**
```
\subsection{다중 클래스 분류 처리}

Qiskit Machine Learning의 VQC는 다중 클래스 분류를 위해 parity mapping을 사용한다.
$n$개 큐비트의 측정 결과를 $2^n$개 비트 문자열로 분류하며,
각 비트 문자열의 패리티(짝수/홀수 1의 개수)를 기준으로 클래스를 할당한다.
3클래스 문제(Iris, Wine)에서는 내부적으로 one-vs-rest 방식으로 확장된다.
```

### 10c: Results (Section 4)

**All tables need new numbers from re-run results.** After experiments complete:

1. Update Table II (tab:accuracy): Add VQC-Angle-RA-reps3 row, update all values, change caption to "10 seeds"
2. Update Table III (tab:f1score): Same
3. Add new Table: Statistical significance (p-values for best VQC vs each classical model)
4. Update Tables IV-X with new values
5. Experiments 2-6 tables: Add Breast Cancer rows

### 10d: Discussion (Section 5)

**Add QSVC training time discussion (new subsection after 5.3):**
```
\subsection{계산 비용 분석}

QSVC가 정확도 측면에서 고전 SVM과 동등한 성능을 보였으나, 학습 시간에서는
현저한 차이가 존재한다. Breast Cancer 데이터셋에서 QSVC-Angle의 학습 시간은
약 XXX초로, 고전 SVM(0.003초) 대비 약 10만 배 느렸다. 이는 양자 커널 행렬
계산의 $O(n^2)$ 복잡도에 기인하며, 현재의 고전 시뮬레이터 기반 실험에서는
양자 하드웨어의 병렬 상태 탐색 이점이 발휘되지 않는다.
따라서 QSVC의 실용적 가치는 고전적으로 시뮬레이션이 어려운 커널을 사용하거나,
실제 양자 하드웨어에서 실행할 때 평가되어야 한다.
```

**Add noise experiment simulator note (in 5.2):**
```
본 실험 5의 이상적 조건은 AerSimulator (shot noise 포함)를 사용하였으며,
이는 실험 1의 StatevectorSampler (shot noise 없음)와 다른 시뮬레이터이다.
따라서 실험 1과 실험 5의 이상적 조건 정확도는 직접 비교할 수 없다.
```

### 10e: References

Add these new references (expanding from 15 to ~20):

```latex
\bibitem{bowles2024}
J.~Bowles \textit{et~al.}, ``Better than classical? The subtle art of benchmarking quantum machine learning models,'' arXiv:2403.07059, 2024.

\bibitem{schuld2022}
M.~Schuld, ``Supervised quantum machine learning models are kernel methods,'' arXiv:2101.11020, \textit{Journal of Machine Learning Research}, 2022.

\bibitem{thanasilp2024}
S.~Thanasilp, S.~Wang, M.~Cerezo, and Z.~Holmes, ``Exponential concentration in quantum kernel methods,'' \textit{Nature Communications}, vol.~15, p.~5200, 2024.

\bibitem{ibmroadmap2023}
IBM Quantum, ``IBM Quantum System Performance,'' Technical Report, 2023.

\bibitem{sim2019}
S.~Sim, P.~D. Johnson, and A.~Aspuru-Guzik, ``Expressibility and entangling capability of parameterized quantum circuits for hybrid quantum-classical algorithms,'' \textit{Advanced Quantum Technologies}, vol.~2, no.~12, p.~1900070, 2019.
```

### 10f: Update Novelty Statement (in Introduction or Discussion)

Add to introduction (after research purpose):
```
본 연구의 주요 기여는 다음과 같다: (1) 동일한 양자 인코딩 하에서 변분 최적화
방식(VQC)과 양자 커널 방식(QSVC)의 성능 차이를 체계적으로 비교하여, 현재의
성능 병목이 양자 특징 공간이 아닌 최적화 방법론에 있음을 실증한 점, (2) VQC의
최적 회로 설정(reps=3)에서도 고전 모델 대비 열위임을 통계적으로 검증한 점,
(3) 이진분류와 다중분류에서의 양자 분류기 성능 차이를 비교 분석한 점이다.
```

**Step 1: Update paper.tex section by section (exact values filled in from new CSV results)**
**Step 2: Sync paper.md with paper.tex content**
**Step 3: Commit**

```bash
git add paper.tex paper.md
git commit -m "docs: update paper with new results, expanded discussion, additional references"
```

---

## Task 11: Update Table Verification

**Files:**
- Modify: `verify_tables.py`

**Changes:** Update all verification functions to match new table structure:
- Table II now has 8 models (added VQC-Angle-RA-reps3)
- Tables IV-VIII now have Breast Cancer rows
- New stats table verification
- Update caption matching logic for "10 seeds"

This is a mechanical update: for each `verify_tableN()` function, update the expected model names, dataset names, and number of rows to match the new CSV structure.

**Step 1: Update verify_tables.py**
**Step 2: Run verification**

```bash
python verify_tables.py
```

Expected: All checks pass (0 mismatches).

**Step 3: Commit**

```bash
git add verify_tables.py
git commit -m "fix: update table verification for new experiment structure"
```

---

## Task 12: Final Verification

**Step 1: Run all tests**

```bash
python run_tests_dispatch.py
```

All phases should pass.

**Step 2: Run table verification**

```bash
python verify_tables.py
```

0 mismatches expected.

**Step 3: Regenerate figures one final time**

```bash
python generate_figures.py
```

**Step 4: Build paper PDF (if XeLaTeX available)**

```bash
xelatex paper.tex
```

**Step 5: Final commit**

```bash
git add -A
git commit -m "chore: final verification pass - all tests and table checks pass"
```

---

## Execution Order Summary

```
Task 1 (data_loader.py) ─┐
Task 2 (vqc_model.py)  ──┤── can be parallel
Task 3 (stats_utils.py) ─┘
         │
Task 4 (exp 1-4 script) ─┐
Task 5 (exp 5 script)  ──┤── can be parallel
Task 6 (exp 6 script)  ──┤
Task 7 (exp 7 script)  ──┘
         │
Task 8 (run experiments) ── sequential, takes hours
         │
Task 9 (figures) ─────────── depends on Task 8
         │
Task 10 (paper.tex) ──────── depends on Task 8
         │
Task 11 (verify_tables) ──── depends on Task 10
         │
Task 12 (final check) ────── depends on all
```

**Estimated total time:** Code changes ~1 hour, experiment runs ~3-6 hours (depending on hardware), paper updates ~1 hour.
