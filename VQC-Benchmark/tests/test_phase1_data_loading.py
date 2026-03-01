"""Phase 1: Data Loading — 데이터셋 로드 기능 단위 테스트"""
import numpy as np
import pytest

PHASE_NAME = "Phase 1: Data Loading"


def test_iris_loads_correct_shape():
    """Iris 데이터 로드: 올바른 shape (150 samples, 4 features → train/test split)"""
    from src.data_loader import load_dataset
    X_train, X_test, y_train, y_test = load_dataset("iris", test_size=0.2, random_state=42)
    assert X_train.shape[0] == 120, f"Train: expected 120, got {X_train.shape[0]}"
    assert X_test.shape[0] == 30, f"Test: expected 30, got {X_test.shape[0]}"
    assert X_train.shape[1] == 4, f"Features: expected 4, got {X_train.shape[1]}"


def test_wine_loads_correct_shape():
    """Wine 데이터 로드: 올바른 shape (178 samples, 13 features)"""
    from src.data_loader import load_dataset
    X_train, X_test, y_train, y_test = load_dataset("wine", test_size=0.2, random_state=42)
    assert X_train.shape[0] + X_test.shape[0] == 178
    assert X_train.shape[1] == 13


def test_breast_cancer_loads():
    """Breast Cancer 데이터 로드: 올바른 shape (569 samples, 30 features)"""
    from src.data_loader import load_dataset
    X_train, X_test, y_train, y_test = load_dataset("breast_cancer", test_size=0.2, random_state=42)
    assert X_train.shape[0] + X_test.shape[0] == 569
    assert X_train.shape[1] == 30


def test_pca_reduces_features():
    """PCA 차원 축소: n_features=4 지정 시 4개 특성으로 축소됨"""
    from src.data_loader import load_dataset
    X_train, X_test, _, _ = load_dataset("wine", n_features=4, random_state=42)
    assert X_train.shape[1] == 4, f"Expected 4 features after PCA, got {X_train.shape[1]}"
    assert X_test.shape[1] == 4


def test_scaling_range():
    """MinMaxScaler [0, pi] 범위: 모든 값이 0 이상 pi 이하"""
    from src.data_loader import load_dataset
    X_train, X_test, _, _ = load_dataset("iris", random_state=42)
    assert X_train.min() >= 0 - 1e-9, f"Min value {X_train.min()} below 0"
    assert X_train.max() <= np.pi + 1e-9, f"Max value {X_train.max()} above pi"
    assert X_test.min() >= -0.5, "Test set values too far below 0"
    assert X_test.max() <= np.pi + 0.5, "Test set values too far above pi"


def test_stratified_split():
    """Stratified Split: train/test에 모든 클래스가 포함됨"""
    from src.data_loader import load_dataset
    _, _, y_train, y_test = load_dataset("iris", random_state=42)
    train_classes = set(np.unique(y_train))
    test_classes = set(np.unique(y_test))
    assert train_classes == test_classes, (
        f"Train classes {train_classes} != Test classes {test_classes}"
    )


def test_unknown_dataset_raises():
    """잘못된 데이터셋 이름: KeyError 발생"""
    from src.data_loader import load_dataset
    with pytest.raises(KeyError):
        load_dataset("nonexistent_dataset")


def test_subsample_dataset():
    """subsample_dataset: fraction=0.5로 데이터 절반 추출"""
    from src.data_loader import load_dataset, subsample_dataset
    X_train, _, y_train, _ = load_dataset("iris", random_state=42)
    X_sub, y_sub = subsample_dataset(X_train, y_train, fraction=0.5, random_state=42)
    assert len(X_sub) < len(X_train), "Subsampled data should be smaller"
    assert len(X_sub) == len(y_sub), "X and y length mismatch"
    assert set(np.unique(y_sub)) == set(np.unique(y_train)), "Missing classes after subsample"
