"""
Experiment 7: QSVC (Quantum Support Vector Classifier) Comparison
FidelityQuantumKernel을 사용한 양자 커널 SVM과 기존 모델들을 비교한다.
3개 데이터셋, 10 seeds per setting.
"""
import sys
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="No gradient function provided")

sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("results", exist_ok=True)

import time
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.circuit.library import ZZFeatureMap
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms.utils import algorithm_globals
from qiskit_machine_learning.kernels import FidelityQuantumKernel

from src.data_loader import load_dataset

SEEDS = [42, 123, 456, 0, 1, 7, 13, 21, 77, 99]


def create_angle_feature_map(n_qubits):
    """Angle encoding feature map."""
    params = ParameterVector("x", n_qubits)
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.ry(params[i], i)
    return qc


def train_qsvc(X_train, y_train, X_test, y_test, n_qubits=4, encoding="angle", random_seed=None):
    """QSVC: 양자 커널을 계산하여 고전 SVM에 사용한다."""
    if random_seed is not None:
        algorithm_globals.random_seed = random_seed

    if encoding == "angle":
        feature_map = create_angle_feature_map(n_qubits)
    elif encoding == "zz":
        feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=1)
    else:
        raise ValueError(f"Unknown encoding: {encoding}")

    sampler = StatevectorSampler()
    kernel = FidelityQuantumKernel(feature_map=feature_map)

    start = time.time()

    # Compute quantum kernel matrices
    kernel_train = kernel.evaluate(X_train)
    kernel_test = kernel.evaluate(X_test, X_train)

    # Train classical SVM with precomputed quantum kernel
    svc = SVC(kernel="precomputed")
    svc.fit(kernel_train, y_train)

    train_time = time.time() - start

    y_pred_train = svc.predict(kernel_train)
    y_pred_test = svc.predict(kernel_test)

    return {
        "train_accuracy": accuracy_score(y_train, y_pred_train),
        "test_accuracy": accuracy_score(y_test, y_pred_test),
        "train_f1": f1_score(y_train, y_pred_train, average="macro"),
        "test_f1": f1_score(y_test, y_pred_test, average="macro"),
        "train_time": train_time,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("EXPERIMENT 7: QSVC Comparison (10 seeds)")
    print("=" * 60)

    datasets = {
        "Iris": {"name": "iris", "n_features": 4},
        "Wine": {"name": "wine", "n_features": 4},
        "Breast Cancer": {"name": "breast_cancer", "n_features": 4},
    }

    qsvc_configs = [
        {"encoding": "angle", "label": "QSVC-Angle"},
        {"encoding": "zz", "label": "QSVC-ZZ"},
    ]

    all_results = []

    for ds_label, ds_info in datasets.items():
        print(f"\n--- Dataset: {ds_label} ---")

        for cfg in qsvc_configs:
            print(f"  Running {cfg['label']} ({len(SEEDS)} seeds)...", end=" ", flush=True)
            seed_results = []

            for seed in SEEDS:
                np.random.seed(seed)
                X_train, X_test, y_train, y_test = load_dataset(
                    ds_info["name"], n_features=ds_info["n_features"], random_state=seed
                )
                try:
                    result = train_qsvc(
                        X_train, y_train, X_test, y_test,
                        n_qubits=ds_info["n_features"],
                        encoding=cfg["encoding"],
                        random_seed=seed,
                    )
                    seed_results.append(result)
                    print(f"s{seed}={result['test_accuracy']:.3f}", end=" ", flush=True)
                except Exception as e:
                    print(f"s{seed} FAILED: {e}", end=" ")

            if seed_results:
                avg = {}
                for key in seed_results[0]:
                    vals = [r[key] for r in seed_results]
                    avg[key] = np.mean(vals)
                    avg[f"{key}_std"] = np.std(vals)
                avg["model"] = cfg["label"]
                avg["dataset"] = ds_label
                avg["n_seeds"] = len(seed_results)
                all_results.append(avg)
                print(f"-> Avg={avg['test_accuracy']:.3f}+/-{avg['test_accuracy_std']:.3f}")

    df = pd.DataFrame(all_results)
    df.to_csv("results/experiment7_qsvc.csv", index=False)
    print(f"\nSaved {len(df)} results to results/experiment7_qsvc.csv")

    # Print summary
    print("\n--- Summary ---")
    for _, row in df.iterrows():
        acc = row['test_accuracy']
        std = row.get('test_accuracy_std', 0)
        print(f"  {row['model']:>12s} | {row['dataset']:>14s} | Acc={acc:.3f}+/-{std:.3f}")
