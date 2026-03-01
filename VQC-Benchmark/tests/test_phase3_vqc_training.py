"""Phase 3: VQC Training — 양자 변분 분류기 단위 테스트"""
import numpy as np
import pytest
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

PHASE_NAME = "Phase 3: VQC Training"

# Inline fixture: 10 samples, 4 features, 3 classes (mimics Iris)
MOCK_X_TRAIN = np.array([
    [0.1, 0.5, 1.0, 0.3], [0.2, 0.6, 1.1, 0.4],
    [0.8, 1.2, 0.5, 1.5], [0.9, 1.3, 0.6, 1.6],
    [1.5, 2.0, 2.5, 2.8], [1.6, 2.1, 2.6, 2.9],
    [0.15, 0.55, 1.05, 0.35], [0.85, 1.25, 0.55, 1.55],
    [1.55, 2.05, 2.55, 2.85], [0.3, 0.7, 1.2, 0.5],
])
MOCK_Y_TRAIN = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0])
MOCK_X_TEST = np.array([
    [0.25, 0.65, 1.15, 0.45],
    [0.95, 1.35, 0.65, 1.65],
    [1.65, 2.15, 2.65, 2.95],
])
MOCK_Y_TEST = np.array([0, 1, 2])


def test_create_feature_map_angle():
    """Angle 인코딩 feature map: 올바른 qubit 수로 생성됨"""
    from src.vqc_model import create_feature_map
    fm = create_feature_map("angle", n_qubits=4)
    assert fm.num_qubits == 4, f"Expected 4 qubits, got {fm.num_qubits}"


def test_create_feature_map_zz():
    """ZZ 인코딩 feature map: 올바른 qubit 수로 생성됨"""
    from src.vqc_model import create_feature_map
    fm = create_feature_map("zz", n_qubits=4)
    assert fm.num_qubits == 4, f"Expected 4 qubits, got {fm.num_qubits}"


def test_create_feature_map_invalid():
    """잘못된 인코딩 이름: ValueError 발생"""
    from src.vqc_model import create_feature_map
    with pytest.raises(ValueError, match="Unknown encoding"):
        create_feature_map("invalid_encoding", n_qubits=4)


def test_create_ansatz_real_amplitudes():
    """RealAmplitudes ansatz: 올바른 qubit 수와 reps"""
    from src.vqc_model import create_ansatz
    ansatz = create_ansatz("real_amplitudes", n_qubits=4, reps=1)
    assert ansatz.num_qubits == 4


def test_create_ansatz_efficient_su2():
    """EfficientSU2 ansatz: 올바른 qubit 수와 reps"""
    from src.vqc_model import create_ansatz
    ansatz = create_ansatz("efficient_su2", n_qubits=4, reps=1)
    assert ansatz.num_qubits == 4


def test_create_ansatz_invalid():
    """잘못된 ansatz 이름: ValueError 발생"""
    from src.vqc_model import create_ansatz
    with pytest.raises(ValueError, match="Unknown ansatz"):
        create_ansatz("invalid_ansatz", n_qubits=4)


def test_train_vqc_returns_metrics():
    """VQC 학습 실행 (maxiter=1): 메트릭 딕셔너리 반환 확인"""
    from src.vqc_model import train_vqc
    result = train_vqc(
        MOCK_X_TRAIN, MOCK_Y_TRAIN, MOCK_X_TEST, MOCK_Y_TEST,
        n_qubits=4, encoding="angle", ansatz_name="real_amplitudes",
        optimizer_name="cobyla", maxiter=1, reps=1,
    )
    required_keys = ["train_accuracy", "test_accuracy", "train_f1", "test_f1", "train_time"]
    for key in required_keys:
        assert key in result, f"Missing key: {key}"
        assert isinstance(result[key], float), f"{key} should be float, got {type(result[key])}"


def test_train_vqc_accuracy_range():
    """VQC 정확도 범위: 0.0 ~ 1.0 사이의 값"""
    from src.vqc_model import train_vqc
    result = train_vqc(
        MOCK_X_TRAIN, MOCK_Y_TRAIN, MOCK_X_TEST, MOCK_Y_TEST,
        n_qubits=4, encoding="angle", ansatz_name="real_amplitudes",
        optimizer_name="cobyla", maxiter=1, reps=1,
    )
    assert 0.0 <= result["test_accuracy"] <= 1.0
    assert 0.0 <= result["train_accuracy"] <= 1.0


def test_train_vqc_positive_time():
    """VQC 학습 시간: 양수"""
    from src.vqc_model import train_vqc
    result = train_vqc(
        MOCK_X_TRAIN, MOCK_Y_TRAIN, MOCK_X_TEST, MOCK_Y_TEST,
        n_qubits=4, encoding="angle", ansatz_name="real_amplitudes",
        optimizer_name="cobyla", maxiter=1, reps=1,
    )
    assert result["train_time"] > 0, "Training time should be positive"
