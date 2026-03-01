"""
Experiment 6: Convergence Analysis
maxiter = 200, 500, 1000으로 VQC 학습 수렴 곡선을 추적한다.
각 iteration마다 objective function 값을 기록하여 수렴 여부를 분석한다.
10 seeds per setting.
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
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals
from qiskit_machine_learning.algorithms import VQC

from src.data_loader import load_dataset

SEEDS = [42, 123, 456, 0, 1, 7, 13, 21, 77, 99]
MAXITERS = [200, 500, 1000]

DATASETS = {
    "Iris": {"name": "iris", "n_features": 4},
    "Breast Cancer": {"name": "breast_cancer", "n_features": 4},
}


class TrackingCOBYLA(COBYLA):
    """COBYLA 옵티마이저를 래핑하여 매 함수 평가마다 loss를 기록한다."""
    def __init__(self, maxiter, loss_history):
        super().__init__(maxiter=maxiter)
        self.loss_history = loss_history

    def minimize(self, fun, x0, jac=None, bounds=None):
        eval_count = [0]
        def tracked_fun(x):
            val = fun(x)
            self.loss_history.append(float(val))
            eval_count[0] += 1
            return val
        result = super().minimize(tracked_fun, x0, jac, bounds)
        return result


def train_vqc_with_callback(X_train, y_train, X_test, y_test, maxiter=200, random_seed=None):
    """VQC를 학습하면서 각 function evaluation의 loss를 기록한다."""
    if random_seed is not None:
        algorithm_globals.random_seed = random_seed

    n_qubits = X_train.shape[1]

    params = ParameterVector("x", n_qubits)
    feature_map = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        feature_map.ry(params[i], i)

    ansatz = RealAmplitudes(num_qubits=n_qubits, reps=1)

    loss_history = []
    optimizer = TrackingCOBYLA(maxiter=maxiter, loss_history=loss_history)
    sampler = StatevectorSampler()
    vqc = VQC(
        sampler=sampler,
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
    )

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
        "loss_history": loss_history,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("EXPERIMENT 6: Convergence Analysis (10 seeds)")
    print("=" * 60)

    all_results = []
    all_curves = {}

    for ds_label, ds_info in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_label}")
        print(f"{'='*60}")

        for maxiter in MAXITERS:
            print(f"\n--- maxiter={maxiter} ({len(SEEDS)} seeds) ---")

            seed_results = []
            seed_curves = []

            for seed in SEEDS:
                np.random.seed(seed)
                X_train, X_test, y_train, y_test = load_dataset(
                    ds_info["name"], n_features=ds_info["n_features"], random_state=seed
                )
                print(f"  seed={seed}...", end=" ", flush=True)
                try:
                    result = train_vqc_with_callback(
                        X_train, y_train, X_test, y_test,
                        maxiter=maxiter, random_seed=seed,
                    )
                    loss_curve = result.pop("loss_history")
                    seed_results.append(result)
                    seed_curves.append(loss_curve)
                    print(f"Acc={result['test_accuracy']:.3f} ({len(loss_curve)} iters)", end=" ")
                except Exception as e:
                    print(f"FAILED: {e}", end=" ")
            print()

            if seed_results:
                avg = {}
                for key in seed_results[0]:
                    vals = [r[key] for r in seed_results]
                    avg[key] = np.mean(vals)
                    avg[f"{key}_std"] = np.std(vals)
                avg["model"] = f"VQC-maxiter{maxiter}"
                avg["dataset"] = ds_label
                avg["maxiter"] = maxiter
                avg["n_seeds"] = len(seed_results)
                avg["n_actual_iters"] = np.mean([len(c) for c in seed_curves])
                all_results.append(avg)
                print(f"  Avg: Acc={avg['test_accuracy']:.3f}+/-{avg['test_accuracy_std']:.3f} "
                      f"Time={avg['train_time']:.1f}s")

                # Store curves keyed by dataset_maxiter
                all_curves[f"{ds_label}_{maxiter}"] = seed_curves

    # Save summary results
    df = pd.DataFrame(all_results)
    df.to_csv("results/experiment6_convergence.csv", index=False)
    print(f"\nSaved summary to results/experiment6_convergence.csv")

    # Save detailed loss curves as JSON
    curves_serializable = {}
    for key, curves in all_curves.items():
        curves_serializable[key] = [
            [float(v) for v in curve] for curve in curves
        ]
    with open("results/experiment6_loss_curves.json", "w") as f:
        json.dump(curves_serializable, f)
    print(f"Saved loss curves to results/experiment6_loss_curves.json")
