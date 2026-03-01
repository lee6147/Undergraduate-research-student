"""Phase 2: Preprocessing — 전처리 파이프라인 단위 테스트"""
import numpy as np
import pytest

PHASE_NAME = "Phase 2: Preprocessing"


def test_minmax_scaling_to_pi():
    """MinMaxScaler 적용: 학습 데이터의 min=0, max=pi"""
    from src.data_loader import load_dataset
    X_train, _, _, _ = load_dataset("iris", random_state=42)
    assert abs(X_train.min()) < 1e-6, f"Min should be ~0, got {X_train.min()}"
    assert abs(X_train.max() - np.pi) < 1e-6, f"Max should be ~pi, got {X_train.max()}"


def test_pca_preserves_sample_count():
    """PCA 적용 후 샘플 수 유지: 행 수 동일"""
    from src.data_loader import load_dataset
    X_train_full, _, _, _ = load_dataset("wine", random_state=42)
    X_train_pca, _, _, _ = load_dataset("wine", n_features=4, random_state=42)
    assert X_train_full.shape[0] == X_train_pca.shape[0], "Sample count changed after PCA"


def test_pca_then_rescale_to_pi():
    """PCA 후 재스케일링: PCA 적용 후에도 [0, pi] 범위 유지"""
    from src.data_loader import load_dataset
    X_train, _, _, _ = load_dataset("wine", n_features=4, random_state=42)
    assert X_train.min() >= -1e-6, f"Min below 0: {X_train.min()}"
    assert X_train.max() <= np.pi + 1e-6, f"Max above pi: {X_train.max()}"


def test_different_seeds_produce_different_splits():
    """다른 seed: 다른 train/test 분할 생성"""
    from src.data_loader import load_dataset
    X1, _, _, _ = load_dataset("iris", random_state=42)
    X2, _, _, _ = load_dataset("iris", random_state=123)
    assert not np.allclose(X1, X2), "Different seeds should produce different splits"


def test_test_set_not_fitted():
    """Test set 독립성: test set 값이 반드시 [0, pi]에 완전히 속하지 않을 수 있음"""
    from src.data_loader import load_dataset
    _, X_test, _, _ = load_dataset("iris", random_state=42)
    assert X_test.min() >= -1.0, f"Test min unreasonably low: {X_test.min()}"
    assert X_test.max() <= np.pi + 1.0, f"Test max unreasonably high: {X_test.max()}"
