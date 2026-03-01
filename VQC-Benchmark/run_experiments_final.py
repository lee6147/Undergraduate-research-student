"""
VQC Benchmark Final: 다중 시드 실행으로 통계적 신뢰성 확보
maxiter=200, 10 seeds per experiment
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

import numpy as np
import pandas as pd
from src.data_loader import load_dataset, subsample_dataset
from src.classical_models import train_classical_model
from src.vqc_model import train_vqc
from src.stats_utils import paired_statistical_test

MAXITER = 200
SEEDS = [42, 123, 456, 0, 1, 7, 13, 21, 77, 99]

DATASETS_EXP234 = {
    "Iris": {"name": "iris", "n_features": 4},
    "Breast Cancer": {"name": "breast_cancer", "n_features": 4},
}


def experiment_1_model_comparison():
    print("=" * 60)
    print("EXPERIMENT 1: Model Comparison (maxiter=200, 10 seeds)")
    print("=" * 60)

    datasets = {
        "Iris": {"name": "iris", "n_features": 4},
        "Wine": {"name": "wine", "n_features": 4},
        "Breast Cancer": {"name": "breast_cancer", "n_features": 4},
    }

    vqc_configs = [
        {"encoding": "angle", "ansatz_name": "real_amplitudes", "label": "VQC-Angle-RA", "reps": 1},
        {"encoding": "angle", "ansatz_name": "efficient_su2", "label": "VQC-Angle-ESU2", "reps": 1},
        {"encoding": "zz", "ansatz_name": "real_amplitudes", "label": "VQC-ZZ-RA", "reps": 1},
        {"encoding": "zz", "ansatz_name": "efficient_su2", "label": "VQC-ZZ-ESU2", "reps": 1},
        {"encoding": "angle", "ansatz_name": "real_amplitudes", "label": "VQC-Angle-RA-reps3", "reps": 3},
    ]

    classical_models = ["SVM", "MLP", "RF"]
    all_results = []
    # Collect per-seed scores for statistical testing
    per_seed_scores = {}  # key: (model, dataset) -> list of test_accuracy

    for ds_label, ds_info in datasets.items():
        print(f"\n--- Dataset: {ds_label} ---")

        for cfg in vqc_configs:
            print(f"  Running {cfg['label']} ({len(SEEDS)} seeds)...", end=" ", flush=True)
            seed_results = []
            seed_accuracies = []
            for seed in SEEDS:
                np.random.seed(seed)
                X_train, X_test, y_train, y_test = load_dataset(
                    ds_info["name"], n_features=ds_info["n_features"], random_state=seed
                )
                try:
                    result = train_vqc(
                        X_train, y_train, X_test, y_test,
                        n_qubits=ds_info["n_features"], encoding=cfg["encoding"],
                        ansatz_name=cfg["ansatz_name"], optimizer_name="cobyla",
                        maxiter=MAXITER, reps=cfg["reps"], random_seed=seed,
                    )
                    seed_results.append(result)
                    seed_accuracies.append(result["test_accuracy"])
                except Exception as e:
                    print(f"seed {seed} FAILED: {e}")

            if seed_results:
                avg_result = {}
                for key in seed_results[0]:
                    vals = [r[key] for r in seed_results]
                    avg_result[key] = np.mean(vals)
                    avg_result[f"{key}_std"] = np.std(vals)
                avg_result["model"] = cfg["label"]
                avg_result["dataset"] = ds_label
                avg_result["n_seeds"] = len(seed_results)
                all_results.append(avg_result)
                per_seed_scores[(cfg["label"], ds_label)] = seed_accuracies
                print(f"Acc={avg_result['test_accuracy']:.3f}±{avg_result['test_accuracy_std']:.3f}")

        for model_name in classical_models:
            print(f"  Running {model_name} ({len(SEEDS)} seeds)...", end=" ", flush=True)
            seed_results = []
            seed_accuracies = []
            for seed in SEEDS:
                np.random.seed(seed)
                X_train, X_test, y_train, y_test = load_dataset(
                    ds_info["name"], n_features=ds_info["n_features"], random_state=seed
                )
                result = train_classical_model(model_name, X_train, y_train, X_test, y_test, random_state=seed)
                seed_results.append(result)
                seed_accuracies.append(result["test_accuracy"])

            avg_result = {}
            for key in seed_results[0]:
                vals = [r[key] for r in seed_results]
                if isinstance(vals[0], (int, float)):
                    avg_result[key] = np.mean(vals)
                    avg_result[f"{key}_std"] = np.std(vals)
                else:
                    avg_result[key] = vals[0]
            avg_result["dataset"] = ds_label
            avg_result["n_seeds"] = len(seed_results)
            all_results.append(avg_result)
            per_seed_scores[(model_name, ds_label)] = seed_accuracies
            print(f"Acc={avg_result['test_accuracy']:.3f}±{avg_result['test_accuracy_std']:.3f}")

    df = pd.DataFrame(all_results)
    df.to_csv("results/experiment1_model_comparison.csv", index=False)
    print(f"\nSaved {len(df)} results")

    # Statistical testing: best VQC vs each classical model per dataset
    stats_results = []
    vqc_labels = [c["label"] for c in vqc_configs]
    for ds_label in datasets:
        # Find best VQC by mean accuracy
        best_vqc = None
        best_acc = -1
        for vl in vqc_labels:
            key = (vl, ds_label)
            if key in per_seed_scores:
                mean_acc = np.mean(per_seed_scores[key])
                if mean_acc > best_acc:
                    best_acc = mean_acc
                    best_vqc = vl

        if best_vqc is None:
            continue

        for cm in classical_models:
            key_cm = (cm, ds_label)
            if key_cm not in per_seed_scores:
                continue
            test_result = paired_statistical_test(
                per_seed_scores[(best_vqc, ds_label)],
                per_seed_scores[key_cm],
            )
            stats_results.append({
                "dataset": ds_label,
                "model_a": best_vqc,
                "model_b": cm,
                "mean_a": np.mean(per_seed_scores[(best_vqc, ds_label)]),
                "mean_b": np.mean(per_seed_scores[key_cm]),
                "test": test_result["test"],
                "statistic": test_result["statistic"],
                "p_value": test_result["p_value"],
                "significant": test_result["significant"],
            })
            print(f"  Stats: {best_vqc} vs {cm} ({ds_label}): p={test_result['p_value']:.4f} {'*' if test_result['significant'] else ''}")

    if stats_results:
        df_stats = pd.DataFrame(stats_results)
        df_stats.to_csv("results/experiment1_stats.csv", index=False)
        print(f"Saved statistical test results")

    return df


def experiment_2_data_size():
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Data Size (10 seeds)")
    print("=" * 60)

    fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
    all_results = []

    for ds_label, ds_info in DATASETS_EXP234.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_label}")
        print(f"{'='*60}")

        for frac in fractions:
            print(f"\n--- Fraction: {frac:.0%} ---")

            # VQC
            print(f"  Running VQC ({len(SEEDS)} seeds)...", end=" ", flush=True)
            seed_results = []
            for seed in SEEDS:
                np.random.seed(seed)
                X_train_full, X_test, y_train_full, y_test = load_dataset(
                    ds_info["name"], n_features=ds_info["n_features"], random_state=seed
                )
                if frac < 1.0:
                    X_train, y_train = subsample_dataset(X_train_full, y_train_full, frac, random_state=seed)
                else:
                    X_train, y_train = X_train_full, y_train_full
                try:
                    result = train_vqc(
                        X_train, y_train, X_test, y_test,
                        n_qubits=ds_info["n_features"], encoding="angle",
                        ansatz_name="real_amplitudes",
                        optimizer_name="cobyla", maxiter=MAXITER, reps=1,
                        random_seed=seed,
                    )
                    result["n_train"] = len(X_train)
                    seed_results.append(result)
                except Exception as e:
                    print(f"seed {seed} FAILED: {e}")

            if seed_results:
                avg = {}
                for key in seed_results[0]:
                    vals = [r[key] for r in seed_results]
                    if isinstance(vals[0], (int, float)):
                        avg[key] = np.mean(vals)
                        avg[f"{key}_std"] = np.std(vals)
                    else:
                        avg[key] = vals[0]
                avg["model"] = "VQC"
                avg["dataset"] = ds_label
                avg["fraction"] = frac
                all_results.append(avg)
                print(f"Acc={avg['test_accuracy']:.3f}±{avg['test_accuracy_std']:.3f}")

            # Classical
            for model_name in ["SVM", "MLP", "RF"]:
                print(f"  Running {model_name} ({len(SEEDS)} seeds)...", end=" ", flush=True)
                seed_results = []
                for seed in SEEDS:
                    np.random.seed(seed)
                    X_train_full, X_test, y_train_full, y_test = load_dataset(
                        ds_info["name"], n_features=ds_info["n_features"], random_state=seed
                    )
                    if frac < 1.0:
                        X_train, y_train = subsample_dataset(X_train_full, y_train_full, frac, random_state=seed)
                    else:
                        X_train, y_train = X_train_full, y_train_full
                    result = train_classical_model(model_name, X_train, y_train, X_test, y_test, random_state=seed)
                    result["n_train"] = len(X_train)
                    seed_results.append(result)

                avg = {}
                for key in seed_results[0]:
                    vals = [r[key] for r in seed_results]
                    if isinstance(vals[0], (int, float)):
                        avg[key] = np.mean(vals)
                        avg[f"{key}_std"] = np.std(vals)
                    else:
                        avg[key] = vals[0]
                avg["dataset"] = ds_label
                avg["fraction"] = frac
                all_results.append(avg)
                print(f"Acc={avg['test_accuracy']:.3f}±{avg['test_accuracy_std']:.3f}")

    df = pd.DataFrame(all_results)
    df.to_csv("results/experiment2_data_size.csv", index=False)
    print(f"\nSaved results")
    return df


def experiment_3_optimizer():
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Optimizer (10 seeds)")
    print("=" * 60)

    all_results = []

    for ds_label, ds_info in DATASETS_EXP234.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_label}")
        print(f"{'='*60}")

        for opt_name in ["cobyla", "spsa"]:
            print(f"\n--- Optimizer: {opt_name.upper()} ({len(SEEDS)} seeds) ---", flush=True)
            seed_results = []
            for seed in SEEDS:
                np.random.seed(seed)
                X_train, X_test, y_train, y_test = load_dataset(
                    ds_info["name"], n_features=ds_info["n_features"], random_state=seed
                )
                print(f"  seed={seed}...", end=" ", flush=True)
                try:
                    result = train_vqc(
                        X_train, y_train, X_test, y_test,
                        n_qubits=ds_info["n_features"], encoding="angle",
                        ansatz_name="real_amplitudes",
                        optimizer_name=opt_name, maxiter=MAXITER, reps=1,
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
                avg["model"] = f"VQC-{opt_name.upper()}"
                avg["dataset"] = ds_label
                avg["optimizer"] = opt_name
                avg["n_seeds"] = len(seed_results)
                all_results.append(avg)
                print(f"  Avg: Acc={avg['test_accuracy']:.3f}±{avg['test_accuracy_std']:.3f} Time={avg['train_time']:.1f}s")

    df = pd.DataFrame(all_results)
    df.to_csv("results/experiment3_optimizer.csv", index=False)
    print(f"\nSaved results")
    return df


def experiment_4_circuit_depth():
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Circuit Depth (10 seeds)")
    print("=" * 60)

    all_results = []

    for ds_label, ds_info in DATASETS_EXP234.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_label}")
        print(f"{'='*60}")

        for reps in [1, 2, 3, 4]:
            print(f"\n--- Reps: {reps} ({len(SEEDS)} seeds) ---", flush=True)
            seed_results = []
            for seed in SEEDS:
                np.random.seed(seed)
                X_train, X_test, y_train, y_test = load_dataset(
                    ds_info["name"], n_features=ds_info["n_features"], random_state=seed
                )
                print(f"  seed={seed}...", end=" ", flush=True)
                try:
                    result = train_vqc(
                        X_train, y_train, X_test, y_test,
                        n_qubits=ds_info["n_features"], encoding="angle",
                        ansatz_name="real_amplitudes",
                        optimizer_name="cobyla", maxiter=MAXITER, reps=reps,
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
                avg["model"] = f"VQC-reps{reps}"
                avg["dataset"] = ds_label
                avg["reps"] = reps
                avg["n_seeds"] = len(seed_results)
                all_results.append(avg)
                print(f"  Avg: Acc={avg['test_accuracy']:.3f}±{avg['test_accuracy_std']:.3f}")

    df = pd.DataFrame(all_results)
    df.to_csv("results/experiment4_circuit_depth.csv", index=False)
    print(f"\nSaved results")
    return df


if __name__ == "__main__":
    print("VQC Benchmark Final - maxiter=200, 10 seeds")
    print("=" * 60)

    df1 = experiment_1_model_comparison()
    df2 = experiment_2_data_size()
    df3 = experiment_3_optimizer()
    df4 = experiment_4_circuit_depth()

    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 60)

    print("\n--- Experiment 1 Summary (mean ± std) ---")
    for _, row in df1.iterrows():
        acc = row['test_accuracy']
        std = row.get('test_accuracy_std', 0)
        print(f"  {row.get('model','?'):>20s} | {row['dataset']:>14s} | Acc={acc:.3f}±{std:.3f}")
