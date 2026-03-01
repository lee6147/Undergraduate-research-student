import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA


DATASETS = {
    "iris": load_iris,
    "wine": load_wine,
    "breast_cancer": load_breast_cancer,
}


def load_dataset(name: str, test_size: float = 0.2, random_state: int = 42, n_features: int = None):
    """데이터셋을 로드하고 전처리하여 train/test로 분할한다.

    PCA가 필요한 경우: StandardScaler → PCA → MinMaxScaler[0,π]
    PCA가 불필요한 경우: MinMaxScaler[0,π]
    """
    loader = DATASETS[name]
    data = loader()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if n_features is not None and n_features < X_train.shape[1]:
        # StandardScaler before PCA for correct variance decomposition
        std_scaler = StandardScaler()
        X_train = std_scaler.fit_transform(X_train)
        X_test = std_scaler.transform(X_test)

        pca = PCA(n_components=n_features)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    # Final scaling to [0, π] for quantum encoding
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def subsample_dataset(X, y, fraction: float, random_state: int = 42):
    """학습 데이터를 일정 비율로 서브샘플링한다."""
    n_samples = max(int(len(X) * fraction), len(np.unique(y)) + 1)
    X_sub, _, y_sub, _ = train_test_split(
        X, y, train_size=n_samples, random_state=random_state, stratify=y
    )
    return X_sub, y_sub
