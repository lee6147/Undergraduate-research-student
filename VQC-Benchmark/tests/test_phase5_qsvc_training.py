"""Phase 5: QSVC Training — 양자 커널 SVM 단위 테스트"""
import sys
import os
import numpy as np
import pytest
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

PHASE_NAME = "Phase 5: QSVC Training"

# Add project root to path for importing run_experiment7_qsvc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Inline fixture: 8 samples, 4 features, 2 classes
MOCK_X_TRAIN = np.array([
    [0.1, 0.3, 0.5, 0.2], [0.2, 0.4, 0.6, 0.3],
    [0.15, 0.35, 0.55, 0.25], [0.25, 0.45, 0.65, 0.35],
    [2.0, 2.2, 2.4, 2.1], [2.1, 2.3, 2.5, 2.2],
    [2.05, 2.25, 2.45, 2.15], [2.15, 2.35, 2.55, 2.25],
])
MOCK_Y_TRAIN = np.array([0, 0, 0, 0, 1, 1, 1, 1])
MOCK_X_TEST = np.array([
    [0.18, 0.38, 0.58, 0.28],
    [2.08, 2.28, 2.48, 2.18],
])
MOCK_Y_TEST = np.array([0, 1])


def test_create_angle_feature_map():
    """Angle feature map 생성: 올바른 qubit 수"""
    from run_experiment7_qsvc import create_angle_feature_map
    fm = create_angle_feature_map(n_qubits=4)
    assert fm.num_qubits == 4


def test_qsvc_angle_returns_metrics():
    """QSVC-Angle 학습: 메트릭 딕셔너리 반환 확인"""
    from run_experiment7_qsvc import train_qsvc
    result = train_qsvc(MOCK_X_TRAIN, MOCK_Y_TRAIN, MOCK_X_TEST, MOCK_Y_TEST,
                         n_qubits=4, encoding="angle")
    required_keys = ["train_accuracy", "test_accuracy", "train_f1", "test_f1", "train_time"]
    for key in required_keys:
        assert key in result, f"Missing key: {key}"


def test_qsvc_zz_returns_metrics():
    """QSVC-ZZ 학습: 메트릭 딕셔너리 반환 확인"""
    from run_experiment7_qsvc import train_qsvc
    result = train_qsvc(MOCK_X_TRAIN, MOCK_Y_TRAIN, MOCK_X_TEST, MOCK_Y_TEST,
                         n_qubits=4, encoding="zz")
    assert 0.0 <= result["test_accuracy"] <= 1.0


def test_qsvc_accuracy_on_separable_data():
    """QSVC 정확도: 분리 가능 데이터에서 높은 정확도"""
    from run_experiment7_qsvc import train_qsvc
    result = train_qsvc(MOCK_X_TRAIN, MOCK_Y_TRAIN, MOCK_X_TEST, MOCK_Y_TEST,
                         n_qubits=4, encoding="angle")
    assert result["test_accuracy"] >= 0.5, f"Expected >= 50%, got {result['test_accuracy']}"


def test_qsvc_invalid_encoding():
    """잘못된 인코딩: ValueError 발생"""
    from run_experiment7_qsvc import train_qsvc
    with pytest.raises(ValueError, match="Unknown encoding"):
        train_qsvc(MOCK_X_TRAIN, MOCK_Y_TRAIN, MOCK_X_TEST, MOCK_Y_TEST,
                    n_qubits=4, encoding="invalid")
