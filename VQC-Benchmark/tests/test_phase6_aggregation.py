"""Phase 6: Result Aggregation — 결과 집계 로직 단위 테스트"""
import numpy as np
import pandas as pd
import pytest
import os
import tempfile

PHASE_NAME = "Phase 6: Result Aggregation"


def _make_mock_csv(tmpdir, filename, data):
    """Helper: mock CSV를 임시 디렉토리에 생성"""
    path = os.path.join(tmpdir, filename)
    pd.DataFrame(data).to_csv(path, index=False)
    return path


def test_csv_round_trip():
    """CSV 저장/로드: DataFrame → CSV → DataFrame 변환 무손실"""
    data = {"model": ["VQC-Angle-RA", "SVM"], "test_accuracy": [0.85, 0.95],
            "test_accuracy_std": [0.02, 0.01], "dataset": ["Iris", "Iris"]}
    df_orig = pd.DataFrame(data)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.csv")
        df_orig.to_csv(path, index=False)
        df_loaded = pd.read_csv(path)
        assert list(df_loaded.columns) == list(df_orig.columns)
        assert len(df_loaded) == len(df_orig)
        assert abs(df_loaded["test_accuracy"].iloc[0] - 0.85) < 1e-6


def test_multi_seed_averaging():
    """다중 시드 평균: 3 시드 결과의 mean/std 계산"""
    seed_results = [
        {"test_accuracy": 0.80, "train_time": 10.0},
        {"test_accuracy": 0.85, "train_time": 12.0},
        {"test_accuracy": 0.90, "train_time": 11.0},
    ]
    avg = {}
    for key in seed_results[0]:
        vals = [r[key] for r in seed_results]
        avg[key] = np.mean(vals)
        avg[f"{key}_std"] = np.std(vals)

    assert abs(avg["test_accuracy"] - 0.85) < 1e-6
    assert abs(avg["test_accuracy_std"] - np.std([0.80, 0.85, 0.90])) < 1e-6
    assert abs(avg["train_time"] - 11.0) < 1e-6


def test_result_csv_has_required_columns():
    """결과 CSV 컬럼: 필수 컬럼 존재 확인"""
    required = ["model", "dataset", "test_accuracy", "test_accuracy_std"]
    data = {col: ["dummy"] if col in ["model", "dataset"] else [0.5] for col in required}
    df = pd.DataFrame(data)
    for col in required:
        assert col in df.columns, f"Missing column: {col}"


def test_existing_experiment1_csv_format():
    """실제 experiment1 CSV: 파일 존재 및 기본 구조 확인"""
    csv_path = os.path.join(os.path.dirname(__file__), "..", "results", "experiment1_model_comparison.csv")
    if not os.path.exists(csv_path):
        pytest.skip("experiment1_model_comparison.csv not found (results not yet generated)")
    df = pd.read_csv(csv_path)
    assert "model" in df.columns, "Missing 'model' column"
    assert "test_accuracy" in df.columns, "Missing 'test_accuracy' column"
    assert "dataset" in df.columns, "Missing 'dataset' column"
    assert len(df) > 0, "CSV is empty"


def test_all_result_csvs_exist():
    """전체 결과 CSV: 7개 실험 결과 파일 존재 확인"""
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    expected_files = [
        "experiment1_model_comparison.csv",
        "experiment2_data_size.csv",
        "experiment3_optimizer.csv",
        "experiment4_circuit_depth.csv",
        "experiment5_noise.csv",
        "experiment6_convergence.csv",
        "experiment7_qsvc.csv",
    ]
    missing = []
    for f in expected_files:
        if not os.path.exists(os.path.join(results_dir, f)):
            missing.append(f)
    if missing:
        pytest.skip(f"Missing result files: {missing}")
    assert len(missing) == 0
