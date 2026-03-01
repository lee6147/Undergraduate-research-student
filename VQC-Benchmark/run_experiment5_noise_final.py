"""
Experiment 5 Final: Noise Impact with 10 seeds for statistical confidence
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
from sklearn.metrics import accuracy_score, f1_score
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals
from qiskit_machine_learning.algorithms import VQC

from src.data_loader import load_dataset

SEEDS = [42, 123, 456, 0, 1, 7, 13, 21, 77, 99]

DATASETS = {
    "Iris": {"name": "iris", "n_features": 4},
    "Breast Cancer": {"name": "breast_cancer", "n_features": 4},
}


def train_vqc_noisy(X_train, y_train, X_test, y_test, error_rate=0.0, maxiter=200, random_seed=None):
    if random_seed is not None:
        algorithm_globals.random_seed = random_seed

    n_qubits = X_train.shape[1]
    params = ParameterVector("x", n_qubits)
    feature_map = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        feature_map.ry(params[i], i)

    ansatz = RealAmplitudes(num_qubits=n_qubits, reps=1)
    optimizer = COBYLA(maxiter=maxiter)

    # Always use AerSimulator for fair comparison (shot noise consistent)
    if error_rate == 0.0:
        backend = AerSimulator()
    else:
        noise_model = NoiseModel()
        error_1q = depolarizing_error(error_rate, 1)
        # 2-qubit gate error ~10x single-qubit on IBM hardware (IBM Quantum roadmap, 2023)
        error_2q = depolarizing_error(min(error_rate * 10, 1.0), 2)
        noise_model.add_all_qubit_quantum_error(error_1q, ['ry', 'rz', 'rx', 'h'])
        noise_model.add_all_qubit_quantum_error(error_2q, ['cx', 'cz'])
        backend = AerSimulator(noise_model=noise_model)

    sampler = AerSampler.from_backend(backend)
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    vqc = VQC(sampler=sampler, feature_map=feature_map, ansatz=ansatz,
               optimizer=optimizer, pass_manager=pm)

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


if __name__ == "__main__":
    print("=" * 60)
    print("EXPERIMENT 5: Noise Impact (10 seeds)")
    print("=" * 60)

    error_rates = [0.0, 0.001, 0.005, 0.01, 0.05]
    all_results = []

    for ds_label, ds_info in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_label}")
        print(f"{'='*60}")

        for error_rate in error_rates:
            label = "Ideal" if error_rate == 0.0 else f"p={error_rate}"
            print(f"\n--- Error Rate: {label} ({len(SEEDS)} seeds) ---")

            seed_results = []
            for seed in SEEDS:
                np.random.seed(seed)
                X_train, X_test, y_train, y_test = load_dataset(
                    ds_info["name"], n_features=ds_info["n_features"], random_state=seed
                )
                print(f"  seed={seed}...", end=" ", flush=True)
                try:
                    result = train_vqc_noisy(
                        X_train, y_train, X_test, y_test,
                        error_rate=error_rate, maxiter=200,
                        random_seed=seed,
                    )
                    seed_results.append(result)
                    print(f"Acc={result['test_accuracy']:.3f}", end=" ")
                except Exception as e:
                    print(f"FAILED: {e}", end=" ")
            print()

            if seed_results:
                avg = {}
                for key in seed_results[0]:
                    vals = [r[key] for r in seed_results]
                    avg[key] = np.mean(vals)
                    avg[f"{key}_std"] = np.std(vals)
                avg["model"] = f"VQC-{label}"
                avg["dataset"] = ds_label
                avg["error_rate"] = error_rate
                avg["n_seeds"] = len(seed_results)
                all_results.append(avg)
                print(f"  Avg: Acc={avg['test_accuracy']:.3f}±{avg['test_accuracy_std']:.3f}")

    df = pd.DataFrame(all_results)
    df.to_csv("results/experiment5_noise.csv", index=False)
    print(f"\nSaved results to results/experiment5_noise.csv")
