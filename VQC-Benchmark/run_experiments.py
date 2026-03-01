"""
VQC Benchmark: 전체 실험 실행 스크립트
양자 변분 분류기 vs 고전 ML 모델 성능 비교
"""
import sys
import os
import json
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from src.data_loader import load_dataset, subsample_dataset
from src.classical_models import train_classical_model
from src.vqc_model import train_vqc

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ── 실험 1: 데이터셋별 모든 모델 비교 ──────────────────────────────
def experiment_1_model_comparison():
    """3개 데이터셋 × (VQC 설정 + 고전 모델) 전체 비교"""
    print("=" * 60)
    print("EXPERIMENT 1: Model Comparison Across Datasets")
    print("=" * 60)

    datasets = {
        "Iris": {"name": "iris", "n_features": 4},
        "Wine": {"name": "wine", "n_features": 4},
        "Breast Cancer": {"name": "breast_cancer", "n_features": 4},
    }

    vqc_configs = [
        {"encoding": "angle", "ansatz_name": "real_amplitudes", "label": "VQC-Angle-RA"},
        {"encoding": "angle", "ansatz_name": "efficient_su2", "label": "VQC-Angle-ESU2"},
        {"encoding": "zz", "ansatz_name": "real_amplitudes", "label": "VQC-ZZ-RA"},
        {"encoding": "zz", "ansatz_name": "efficient_su2", "label": "VQC-ZZ-ESU2"},
    ]

    classical_models = ["SVM", "MLP", "RF"]
    all_results = []

    for ds_label, ds_info in datasets.items():
        print(f"\n--- Dataset: {ds_label} ---")
        X_train, X_test, y_train, y_test = load_dataset(
            ds_info["name"], n_features=ds_info["n_features"]
        )
        n_qubits = ds_info["n_features"]

        # VQC 실험
        for cfg in vqc_configs:
            print(f"  Running {cfg['label']}...", end=" ", flush=True)
            try:
                result = train_vqc(
                    X_train, y_train, X_test, y_test,
                    n_qubits=n_qubits, encoding=cfg["encoding"],
                    ansatz_name=cfg["ansatz_name"], optimizer_name="cobyla",
                    maxiter=50, reps=1,
                )
                result["model"] = cfg["label"]
                result["dataset"] = ds_label
                all_results.append(result)
                print(f"Acc={result['test_accuracy']:.3f} Time={result['train_time']:.1f}s")
            except Exception as e:
                print(f"FAILED: {e}")

        # 고전 모델 실험
        for model_name in classical_models:
            print(f"  Running {model_name}...", end=" ", flush=True)
            result = train_classical_model(model_name, X_train, y_train, X_test, y_test)
            result["dataset"] = ds_label
            all_results.append(result)
            print(f"Acc={result['test_accuracy']:.3f} Time={result['train_time']:.4f}s")

    df = pd.DataFrame(all_results)
    df.to_csv("results/experiment1_model_comparison.csv", index=False)
    print(f"\nSaved {len(df)} results to results/experiment1_model_comparison.csv")
    return df


# ── 실험 2: 데이터 크기별 성능 비교 ──────────────────────────────
def experiment_2_data_size():
    """학습 데이터 비율에 따른 성능 변화 (Iris 데이터셋)"""
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Data Size vs Performance (Iris)")
    print("=" * 60)

    fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
    X_train_full, X_test, y_train_full, y_test = load_dataset("iris", n_features=4)
    n_qubits = 4
    all_results = []

    for frac in fractions:
        print(f"\n--- Fraction: {frac:.0%} ---")

        if frac < 1.0:
            X_train, y_train = subsample_dataset(X_train_full, y_train_full, frac)
        else:
            X_train, y_train = X_train_full, y_train_full

        print(f"  Training samples: {len(X_train)}")

        # VQC (best config from exp1: angle + real_amplitudes)
        print(f"  Running VQC...", end=" ", flush=True)
        try:
            result = train_vqc(
                X_train, y_train, X_test, y_test,
                n_qubits=n_qubits, encoding="angle", ansatz_name="real_amplitudes",
                optimizer_name="cobyla", maxiter=50, reps=1,
            )
            result["model"] = "VQC"
            result["dataset"] = "Iris"
            result["fraction"] = frac
            result["n_train"] = len(X_train)
            all_results.append(result)
            print(f"Acc={result['test_accuracy']:.3f}")
        except Exception as e:
            print(f"FAILED: {e}")

        # 고전 모델
        for model_name in ["SVM", "MLP", "RF"]:
            print(f"  Running {model_name}...", end=" ", flush=True)
            result = train_classical_model(model_name, X_train, y_train, X_test, y_test)
            result["dataset"] = "Iris"
            result["fraction"] = frac
            result["n_train"] = len(X_train)
            all_results.append(result)
            print(f"Acc={result['test_accuracy']:.3f}")

    df = pd.DataFrame(all_results)
    df.to_csv("results/experiment2_data_size.csv", index=False)
    print(f"\nSaved {len(df)} results to results/experiment2_data_size.csv")
    return df


# ── 실험 3: 옵티마이저 비교 ──────────────────────────────
def experiment_3_optimizer():
    """COBYLA vs SPSA 옵티마이저 비교 (Iris)"""
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Optimizer Comparison (Iris)")
    print("=" * 60)

    X_train, X_test, y_train, y_test = load_dataset("iris", n_features=4)
    n_qubits = 4
    all_results = []

    for opt_name in ["cobyla", "spsa"]:
        print(f"\n--- Optimizer: {opt_name.upper()} ---")
        print(f"  Running VQC-{opt_name}...", end=" ", flush=True)
        try:
            result = train_vqc(
                X_train, y_train, X_test, y_test,
                n_qubits=n_qubits, encoding="angle", ansatz_name="real_amplitudes",
                optimizer_name=opt_name, maxiter=50, reps=1,
            )
            result["model"] = f"VQC-{opt_name.upper()}"
            result["dataset"] = "Iris"
            result["optimizer"] = opt_name
            all_results.append(result)
            print(f"Acc={result['test_accuracy']:.3f} Time={result['train_time']:.1f}s")
        except Exception as e:
            print(f"FAILED: {e}")

    df = pd.DataFrame(all_results)
    df.to_csv("results/experiment3_optimizer.csv", index=False)
    print(f"\nSaved results to results/experiment3_optimizer.csv")
    return df


# ── 실험 4: 회로 깊이 비교 ──────────────────────────────
def experiment_4_circuit_depth():
    """회로 깊이(reps)에 따른 성능 변화 (Iris)"""
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Circuit Depth vs Performance (Iris)")
    print("=" * 60)

    X_train, X_test, y_train, y_test = load_dataset("iris", n_features=4)
    n_qubits = 4
    all_results = []

    for reps in [1, 2, 3, 4]:
        print(f"\n--- Reps: {reps} ---")
        print(f"  Running VQC (reps={reps})...", end=" ", flush=True)
        try:
            result = train_vqc(
                X_train, y_train, X_test, y_test,
                n_qubits=n_qubits, encoding="angle", ansatz_name="real_amplitudes",
                optimizer_name="cobyla", maxiter=50, reps=reps,
            )
            result["model"] = f"VQC-reps{reps}"
            result["dataset"] = "Iris"
            result["reps"] = reps
            all_results.append(result)
            print(f"Acc={result['test_accuracy']:.3f} Time={result['train_time']:.1f}s")
        except Exception as e:
            print(f"FAILED: {e}")

    df = pd.DataFrame(all_results)
    df.to_csv("results/experiment4_circuit_depth.csv", index=False)
    print(f"\nSaved results to results/experiment4_circuit_depth.csv")
    return df


# ── Main ──────────────────────────────
if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("VQC Benchmark - Starting All Experiments")
    print("=" * 60)

    df1 = experiment_1_model_comparison()
    df2 = experiment_2_data_size()
    df3 = experiment_3_optimizer()
    df4 = experiment_4_circuit_depth()

    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 60)

    # 요약 출력
    print("\n--- Experiment 1 Summary (Test Accuracy) ---")
    pivot = df1.pivot_table(values="test_accuracy", index="model", columns="dataset")
    print(pivot.to_string(float_format="%.3f"))

    print("\n--- Experiment 2 Summary (Data Size) ---")
    pivot2 = df2.pivot_table(values="test_accuracy", index="fraction", columns="model")
    print(pivot2.to_string(float_format="%.3f"))
