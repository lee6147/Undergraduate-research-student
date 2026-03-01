"""Phase 4: Classical Training — 고전 모델 학습 단위 테스트"""
import numpy as np
import pytest

PHASE_NAME = "Phase 4: Classical Training"

# Inline fixture: linearly separable 2-class data
MOCK_X_TRAIN = np.array([
    [0.1, 0.2], [0.2, 0.3], [0.15, 0.25], [0.3, 0.1],
    [2.0, 2.1], [2.1, 2.2], [2.05, 2.15], [2.2, 2.0],
])
MOCK_Y_TRAIN = np.array([0, 0, 0, 0, 1, 1, 1, 1])
MOCK_X_TEST = np.array([[0.25, 0.15], [2.15, 2.05]])
MOCK_Y_TEST = np.array([0, 1])


def test_svm_returns_metrics():
    """SVM 학습: 메트릭 딕셔너리 반환 확인"""
    from src.classical_models import train_classical_model
    result = train_classical_model("SVM", MOCK_X_TRAIN, MOCK_Y_TRAIN, MOCK_X_TEST, MOCK_Y_TEST)
    required_keys = ["model", "train_accuracy", "test_accuracy", "train_f1", "test_f1", "train_time"]
    for key in required_keys:
        assert key in result, f"Missing key: {key}"


def test_mlp_returns_metrics():
    """MLP 학습: 메트릭 딕셔너리 반환 확인"""
    from src.classical_models import train_classical_model
    result = train_classical_model("MLP", MOCK_X_TRAIN, MOCK_Y_TRAIN, MOCK_X_TEST, MOCK_Y_TEST)
    assert result["model"] == "MLP"
    assert 0.0 <= result["test_accuracy"] <= 1.0


def test_rf_returns_metrics():
    """Random Forest 학습: 메트릭 딕셔너리 반환 확인"""
    from src.classical_models import train_classical_model
    result = train_classical_model("RF", MOCK_X_TRAIN, MOCK_Y_TRAIN, MOCK_X_TEST, MOCK_Y_TEST)
    assert result["model"] == "RF"
    assert 0.0 <= result["test_accuracy"] <= 1.0


def test_svm_high_accuracy_on_separable_data():
    """SVM 정확도: 선형 분리 가능 데이터에서 높은 정확도"""
    from src.classical_models import train_classical_model
    result = train_classical_model("SVM", MOCK_X_TRAIN, MOCK_Y_TRAIN, MOCK_X_TEST, MOCK_Y_TEST)
    assert result["test_accuracy"] == 1.0, f"Expected 100% on separable data, got {result['test_accuracy']}"


def test_training_time_positive():
    """학습 시간: 양수"""
    from src.classical_models import train_classical_model
    result = train_classical_model("SVM", MOCK_X_TRAIN, MOCK_Y_TRAIN, MOCK_X_TEST, MOCK_Y_TEST)
    assert result["train_time"] > 0


def test_random_state_reproducibility():
    """재현성: 동일 seed로 동일 결과"""
    from src.classical_models import train_classical_model
    r1 = train_classical_model("RF", MOCK_X_TRAIN, MOCK_Y_TRAIN, MOCK_X_TEST, MOCK_Y_TEST, random_state=42)
    r2 = train_classical_model("RF", MOCK_X_TRAIN, MOCK_Y_TRAIN, MOCK_X_TEST, MOCK_Y_TEST, random_state=42)
    assert r1["test_accuracy"] == r2["test_accuracy"], "Same seed should produce same accuracy"
