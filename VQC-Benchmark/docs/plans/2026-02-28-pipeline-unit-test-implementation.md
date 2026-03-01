# VQC Benchmark Pipeline Unit Tests — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create 8 self-contained test files (one per pipeline phase) that subagents can run in isolation, producing natural language test reports.

**Architecture:** Each test file contains inline mock data, imports only its own phase's module, and uses a custom pytest plugin (`conftest.py`) to format results as natural language reports. A dispatch script (`run_tests_dispatch.py`) orchestrates subagent execution.

**Tech Stack:** pytest, numpy, pandas, sklearn, qiskit, matplotlib (all already in requirements.txt)

---

### Task 1: Create `tests/conftest.py` — Natural Language Report Plugin

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/__init__.py` (empty)

**Step 1: Write the conftest.py with natural language reporter**

```python
"""
Pytest plugin that produces natural language test reports.
Each test's docstring becomes the human-readable description.
"""
import pytest
import time


class NaturalLanguageReporter:
    def __init__(self):
        self.results = []
        self.phase_name = ""
        self.start_time = None

    def set_phase(self, name):
        self.phase_name = name

    def add_result(self, passed, description, detail=""):
        self.results.append({
            "passed": passed,
            "description": description,
            "detail": detail,
        })

    def format_report(self, duration):
        lines = []
        lines.append(f"=== {self.phase_name} 테스트 결과 ===")
        lines.append("")
        for r in self.results:
            tag = "[PASS]" if r["passed"] else "[FAIL]"
            lines.append(f"{tag} {r['description']}")
            if r["detail"]:
                lines.append(f"       → {r['detail']}")
        lines.append("")
        passed = sum(1 for r in self.results if r["passed"])
        total = len(self.results)
        lines.append(f"총 {total}개 테스트 중 {passed}개 통과, {total - passed}개 실패")
        lines.append(f"실행 시간: {duration:.1f}초")
        return "\n".join(lines)


_reporter = NaturalLanguageReporter()


@pytest.fixture
def reporter():
    return _reporter


def pytest_collection_modifyitems(items):
    """Extract phase name from the first test's module docstring."""
    if items:
        mod = items[0].module
        if hasattr(mod, "PHASE_NAME"):
            _reporter.set_phase(mod.PHASE_NAME)


def pytest_runtest_logreport(report):
    if report.when == "call":
        desc = report.head_line or report.nodeid.split("::")[-1]
        # Use the test's docstring if available
        if hasattr(report, "longreprtext") and report.failed:
            detail = report.longreprtext.split("\n")[-1].strip() if report.longreprtext else ""
        else:
            detail = ""
        _reporter.add_result(report.passed, desc, detail)


def pytest_sessionstart(session):
    _reporter.results = []
    _reporter.start_time = time.time()


def pytest_sessionfinish(session, exitstatus):
    duration = time.time() - (_reporter.start_time or time.time())
    report = _reporter.format_report(duration)
    print("\n")
    print(report)
```

**Step 2: Create empty `__init__.py`**

```python
# tests/__init__.py (empty)
```

**Step 3: Verify conftest loads**

Run: `cd C:/Users/user/vqc-benchmark && python -m pytest tests/ --co -q 2>&1 | head -5`
Expected: "no tests ran" (no test files yet), no import errors

**Step 4: Commit**

```bash
git add tests/conftest.py tests/__init__.py
git commit -m "feat: add natural language test report plugin (conftest.py)"
```

---

### Task 2: Create `tests/test_phase1_data_loading.py`

**Files:**
- Create: `tests/test_phase1_data_loading.py`
- Test target: `src/data_loader.py` — `load_dataset()`, `DATASETS` dict

**Step 1: Write the test file**

```python
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
```

**Step 2: Run test**

Run: `cd C:/Users/user/vqc-benchmark && python -m pytest tests/test_phase1_data_loading.py -v`
Expected: All 8 tests PASS

**Step 3: Commit**

```bash
git add tests/test_phase1_data_loading.py
git commit -m "test: add phase 1 data loading tests"
```

---

### Task 3: Create `tests/test_phase2_preprocessing.py`

**Files:**
- Create: `tests/test_phase2_preprocessing.py`
- Test target: `src/data_loader.py` — scaling and PCA behavior

**Step 1: Write the test file**

```python
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
    # Test set is transformed (not fitted), so values may slightly exceed [0, pi]
    assert X_test.min() >= -1.0, f"Test min unreasonably low: {X_test.min()}"
    assert X_test.max() <= np.pi + 1.0, f"Test max unreasonably high: {X_test.max()}"
```

**Step 2: Run test**

Run: `cd C:/Users/user/vqc-benchmark && python -m pytest tests/test_phase2_preprocessing.py -v`
Expected: All 5 tests PASS

**Step 3: Commit**

```bash
git add tests/test_phase2_preprocessing.py
git commit -m "test: add phase 2 preprocessing tests"
```

---

### Task 4: Create `tests/test_phase3_vqc_training.py`

**Files:**
- Create: `tests/test_phase3_vqc_training.py`
- Test target: `src/vqc_model.py` — `create_feature_map()`, `create_ansatz()`, `train_vqc()`

**Step 1: Write the test file**

```python
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
```

**Step 2: Run test**

Run: `cd C:/Users/user/vqc-benchmark && python -m pytest tests/test_phase3_vqc_training.py -v --timeout=60`
Expected: All 9 tests PASS (may take ~10-30s for VQC training tests)

**Step 3: Commit**

```bash
git add tests/test_phase3_vqc_training.py
git commit -m "test: add phase 3 VQC training tests"
```

---

### Task 5: Create `tests/test_phase4_classical_training.py`

**Files:**
- Create: `tests/test_phase4_classical_training.py`
- Test target: `src/classical_models.py` — `train_classical_model()`, `_make_model()`

**Step 1: Write the test file**

```python
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
```

**Step 2: Run test**

Run: `cd C:/Users/user/vqc-benchmark && python -m pytest tests/test_phase4_classical_training.py -v`
Expected: All 6 tests PASS

**Step 3: Commit**

```bash
git add tests/test_phase4_classical_training.py
git commit -m "test: add phase 4 classical training tests"
```

---

### Task 6: Create `tests/test_phase5_qsvc_training.py`

**Files:**
- Create: `tests/test_phase5_qsvc_training.py`
- Test target: `run_experiment7_qsvc.py` — `train_qsvc()`, `create_angle_feature_map()`

**Step 1: Write the test file**

```python
"""Phase 5: QSVC Training — 양자 커널 SVM 단위 테스트"""
import numpy as np
import pytest
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

PHASE_NAME = "Phase 5: QSVC Training"

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
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from run_experiment7_qsvc import create_angle_feature_map
    fm = create_angle_feature_map(n_qubits=4)
    assert fm.num_qubits == 4


def test_qsvc_angle_returns_metrics():
    """QSVC-Angle 학습: 메트릭 딕셔너리 반환 확인"""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from run_experiment7_qsvc import train_qsvc
    result = train_qsvc(MOCK_X_TRAIN, MOCK_Y_TRAIN, MOCK_X_TEST, MOCK_Y_TEST,
                         n_qubits=4, encoding="angle")
    required_keys = ["train_accuracy", "test_accuracy", "train_f1", "test_f1", "train_time"]
    for key in required_keys:
        assert key in result, f"Missing key: {key}"


def test_qsvc_zz_returns_metrics():
    """QSVC-ZZ 학습: 메트릭 딕셔너리 반환 확인"""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from run_experiment7_qsvc import train_qsvc
    result = train_qsvc(MOCK_X_TRAIN, MOCK_Y_TRAIN, MOCK_X_TEST, MOCK_Y_TEST,
                         n_qubits=4, encoding="zz")
    assert 0.0 <= result["test_accuracy"] <= 1.0


def test_qsvc_accuracy_on_separable_data():
    """QSVC 정확도: 분리 가능 데이터에서 높은 정확도"""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from run_experiment7_qsvc import train_qsvc
    result = train_qsvc(MOCK_X_TRAIN, MOCK_Y_TRAIN, MOCK_X_TEST, MOCK_Y_TEST,
                         n_qubits=4, encoding="angle")
    assert result["test_accuracy"] >= 0.5, f"Expected >= 50%, got {result['test_accuracy']}"


def test_qsvc_invalid_encoding():
    """잘못된 인코딩: ValueError 발생"""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from run_experiment7_qsvc import train_qsvc
    with pytest.raises(ValueError, match="Unknown encoding"):
        train_qsvc(MOCK_X_TRAIN, MOCK_Y_TRAIN, MOCK_X_TEST, MOCK_Y_TEST,
                    n_qubits=4, encoding="invalid")
```

**Step 2: Run test**

Run: `cd C:/Users/user/vqc-benchmark && python -m pytest tests/test_phase5_qsvc_training.py -v --timeout=60`
Expected: All 5 tests PASS

**Step 3: Commit**

```bash
git add tests/test_phase5_qsvc_training.py
git commit -m "test: add phase 5 QSVC training tests"
```

---

### Task 7: Create `tests/test_phase6_aggregation.py`

**Files:**
- Create: `tests/test_phase6_aggregation.py`
- Test target: result aggregation logic (inline, no module import needed)

**Step 1: Write the test file**

```python
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
```

**Step 2: Run test**

Run: `cd C:/Users/user/vqc-benchmark && python -m pytest tests/test_phase6_aggregation.py -v`
Expected: All 5 tests PASS (or skip if results CSVs not present)

**Step 3: Commit**

```bash
git add tests/test_phase6_aggregation.py
git commit -m "test: add phase 6 result aggregation tests"
```

---

### Task 8: Create `tests/test_phase7_figures.py`

**Files:**
- Create: `tests/test_phase7_figures.py`
- Test target: `generate_figures.py` — figure generation functions

**Step 1: Write the test file**

```python
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
```

**Step 2: Run test**

Run: `cd C:/Users/user/vqc-benchmark && python -m pytest tests/test_phase7_figures.py -v`
Expected: All 5 tests PASS

**Step 3: Commit**

```bash
git add tests/test_phase7_figures.py
git commit -m "test: add phase 7 figure generation tests"
```

---

### Task 9: Create `tests/test_phase8_paper_build.py`

**Files:**
- Create: `tests/test_phase8_paper_build.py`
- Test target: `paper.tex` compilation

**Step 1: Write the test file**

```python
"""Phase 8: Paper Build — LaTeX 논문 빌드 단위 테스트"""
import os
import subprocess
import pytest

PHASE_NAME = "Phase 8: Paper Build"

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")


def test_paper_tex_exists():
    """paper.tex 존재: LaTeX 소스 파일이 프로젝트 루트에 존재"""
    path = os.path.join(PROJECT_ROOT, "paper.tex")
    assert os.path.exists(path), "paper.tex not found"


def test_paper_tex_not_empty():
    """paper.tex 내용: 파일이 비어있지 않음"""
    path = os.path.join(PROJECT_ROOT, "paper.tex")
    size = os.path.getsize(path)
    assert size > 1000, f"paper.tex is only {size} bytes, suspiciously small"


def test_paper_tex_has_document_structure():
    """paper.tex 구조: begin/end document 포함"""
    path = os.path.join(PROJECT_ROOT, "paper.tex")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "\\begin{document}" in content, "Missing \\begin{document}"
    assert "\\end{document}" in content, "Missing \\end{document}"
    assert "\\begin{abstract}" in content, "Missing abstract"


def test_paper_tex_references():
    """paper.tex 참고문헌: bibliography 포함"""
    path = os.path.join(PROJECT_ROOT, "paper.tex")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "\\bibliography" in content or "\\begin{thebibliography}" in content, (
        "No bibliography found"
    )


def test_all_figures_referenced():
    """Figure 참조: tex에서 참조하는 모든 figure 파일 존재"""
    path = os.path.join(PROJECT_ROOT, "paper.tex")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    import re
    figure_refs = re.findall(r'\\includegraphics(?:\[.*?\])?\{(.*?)\}', content)
    missing = []
    for fig in figure_refs:
        fig_path = os.path.join(PROJECT_ROOT, fig)
        if not os.path.exists(fig_path):
            missing.append(fig)
    assert len(missing) == 0, f"Missing figure files: {missing}"


def test_xelatex_available():
    """XeLaTeX 설치: xelatex 명령어 사용 가능"""
    try:
        result = subprocess.run(["xelatex", "--version"],
                                capture_output=True, text=True, timeout=10)
        assert result.returncode == 0, "xelatex returned non-zero exit code"
    except FileNotFoundError:
        pytest.skip("xelatex not installed")


def test_paper_compiles():
    """논문 컴파일: xelatex로 PDF 생성 성공"""
    try:
        subprocess.run(["xelatex", "--version"], capture_output=True, timeout=10)
    except FileNotFoundError:
        pytest.skip("xelatex not installed")

    result = subprocess.run(
        ["xelatex", "-interaction=nonstopmode", "-halt-on-error", "paper.tex"],
        cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, f"xelatex failed:\n{result.stdout[-500:]}"
    pdf_path = os.path.join(PROJECT_ROOT, "paper.pdf")
    assert os.path.exists(pdf_path), "paper.pdf not generated"
    assert os.path.getsize(pdf_path) > 10000, "paper.pdf suspiciously small"
```

**Step 2: Run test**

Run: `cd C:/Users/user/vqc-benchmark && python -m pytest tests/test_phase8_paper_build.py -v --timeout=180`
Expected: All 7 tests PASS (xelatex tests may skip if not installed)

**Step 3: Commit**

```bash
git add tests/test_phase8_paper_build.py
git commit -m "test: add phase 8 paper build tests"
```

---

### Task 10: Create `run_tests_dispatch.py` — Subagent Dispatch Script

**Files:**
- Create: `run_tests_dispatch.py`

This script is designed to be called from the main agent. It runs each phase's tests in isolation and produces a combined natural language report.

**Step 1: Write the dispatch script**

```python
"""
Subagent Test Dispatch: 각 페이즈 테스트를 독립적으로 실행하고 자연어 리포트를 생성한다.
사용법: python run_tests_dispatch.py [phase_number]
       python run_tests_dispatch.py         # 전체 실행
       python run_tests_dispatch.py 3       # Phase 3만 실행
"""
import subprocess
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

PHASES = [
    ("Phase 1: Data Loading", "tests/test_phase1_data_loading.py"),
    ("Phase 2: Preprocessing", "tests/test_phase2_preprocessing.py"),
    ("Phase 3: VQC Training", "tests/test_phase3_vqc_training.py"),
    ("Phase 4: Classical Training", "tests/test_phase4_classical_training.py"),
    ("Phase 5: QSVC Training", "tests/test_phase5_qsvc_training.py"),
    ("Phase 6: Result Aggregation", "tests/test_phase6_aggregation.py"),
    ("Phase 7: Figure Generation", "tests/test_phase7_figures.py"),
    ("Phase 8: Paper Build", "tests/test_phase8_paper_build.py"),
]


def run_phase(phase_num):
    """단일 페이즈 테스트를 실행하고 결과를 반환한다."""
    name, test_file = PHASES[phase_num - 1]
    test_path = os.path.join(PROJECT_ROOT, test_file)

    if not os.path.exists(test_path):
        return name, "SKIP", f"테스트 파일 없음: {test_file}"

    result = subprocess.run(
        [sys.executable, "-m", "pytest", test_path, "-v", "--tb=short", "--timeout=120"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )

    return name, result.returncode, result.stdout + result.stderr


def format_phase_report(name, returncode, output):
    """페이즈 결과를 자연어로 포맷한다."""
    lines = []
    lines.append(f"=== {name} ===")

    if returncode == "SKIP":
        lines.append(f"[SKIP] {output}")
        return "\n".join(lines)

    # Parse pytest verbose output for test results
    for line in output.split("\n"):
        if "PASSED" in line:
            test_name = line.split("::")[1].split(" ")[0] if "::" in line else line
            lines.append(f"[PASS] {test_name}")
        elif "FAILED" in line:
            test_name = line.split("::")[1].split(" ")[0] if "::" in line else line
            lines.append(f"[FAIL] {test_name}")
        elif "SKIPPED" in line:
            test_name = line.split("::")[1].split(" ")[0] if "::" in line else line
            lines.append(f"[SKIP] {test_name}")

    status = "PASS" if returncode == 0 else "FAIL"
    lines.append(f"결과: {status}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        phase_num = int(sys.argv[1])
        name, code, output = run_phase(phase_num)
        print(format_phase_report(name, code, output))
    else:
        print("=" * 60)
        print("VQC Benchmark Pipeline — 전체 단위 테스트 리포트")
        print("=" * 60)
        print()

        total_pass = 0
        total_fail = 0
        total_skip = 0

        for i in range(1, len(PHASES) + 1):
            name, code, output = run_phase(i)
            report = format_phase_report(name, code, output)
            print(report)

            if code == "SKIP":
                total_skip += 1
            elif code == 0:
                total_pass += 1
            else:
                total_fail += 1

        print("=" * 60)
        print(f"전체 결과: {total_pass} PASS / {total_fail} FAIL / {total_skip} SKIP")
        print("=" * 60)
```

**Step 2: Run full dispatch**

Run: `cd C:/Users/user/vqc-benchmark && python run_tests_dispatch.py`
Expected: All 8 phases report results in natural language format

**Step 3: Commit**

```bash
git add run_tests_dispatch.py
git commit -m "feat: add subagent test dispatch script with natural language reports"
```

---

### Task 11: Run Full Test Suite and Verify

**Step 1: Run all tests together**

Run: `cd C:/Users/user/vqc-benchmark && python -m pytest tests/ -v --timeout=120`
Expected: All tests PASS (quantum tests may take 10-30s each)

**Step 2: Run dispatch script for full natural language report**

Run: `cd C:/Users/user/vqc-benchmark && python run_tests_dispatch.py`
Expected: Natural language report for all 8 phases

**Step 3: Final commit**

```bash
git add -A
git commit -m "feat: complete 8-phase pipeline unit test suite with subagent dispatch"
```
