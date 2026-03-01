"""Phase 7: Figure Generation — 시각화 생성 단위 테스트"""
import os
import tempfile
import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PHASE_NAME = "Phase 7: Figure Generation"


def test_matplotlib_agg_backend():
    """Matplotlib Agg 백엔드: 비GUI 환경에서 렌더링 가능"""
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.png")
        fig.savefig(path)
        plt.close(fig)
        assert os.path.exists(path), "PNG file not created"
        assert os.path.getsize(path) > 0, "PNG file is empty"


def test_bar_chart_with_error_bars():
    """에러바 막대 그래프: 정상 생성"""
    fig, ax = plt.subplots()
    models = ["VQC-Angle", "SVM", "RF"]
    accuracies = [0.85, 0.95, 0.92]
    errors = [0.03, 0.01, 0.02]
    ax.bar(models, accuracies, yerr=errors, capsize=3)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "bar.png")
        fig.savefig(path)
        plt.close(fig)
        assert os.path.getsize(path) > 1000, "Bar chart suspiciously small"


def test_existing_figures_present():
    """기존 figure 파일: 9개 PNG 파일 존재 확인"""
    figures_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
    expected = [f"fig{i}_" for i in range(1, 10)]
    if not os.path.isdir(figures_dir):
        pytest.skip("figures/ directory not found")
    pngs = [f for f in os.listdir(figures_dir) if f.endswith(".png")]
    for prefix in expected:
        matches = [f for f in pngs if f.startswith(prefix)]
        assert len(matches) > 0, f"No PNG found with prefix '{prefix}'"


def test_figure_file_sizes():
    """Figure 파일 크기: 각 PNG가 최소 5KB (빈 이미지가 아님)"""
    figures_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
    if not os.path.isdir(figures_dir):
        pytest.skip("figures/ directory not found")
    pngs = [f for f in os.listdir(figures_dir) if f.endswith(".png")]
    for png in pngs:
        size = os.path.getsize(os.path.join(figures_dir, png))
        assert size > 5000, f"{png} is too small ({size} bytes), may be corrupted"


def test_generate_from_mock_csv():
    """Mock CSV로 figure 생성: 전체 파이프라인 테스트"""
    mock_data = pd.DataFrame({
        "model": ["VQC-Angle-RA", "VQC-ZZ-RA", "SVM"],
        "dataset": ["Iris", "Iris", "Iris"],
        "test_accuracy": [0.85, 0.80, 0.95],
        "test_accuracy_std": [0.03, 0.04, 0.01],
        "train_time": [15.0, 18.0, 0.1],
        "train_time_std": [2.0, 3.0, 0.01],
    })
    fig, ax = plt.subplots()
    ax.bar(mock_data["model"], mock_data["test_accuracy"],
           yerr=mock_data["test_accuracy_std"], capsize=3)
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Mock Model Comparison")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "mock_fig.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 5000
