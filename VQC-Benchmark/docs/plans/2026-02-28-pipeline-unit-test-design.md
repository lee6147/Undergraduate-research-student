# VQC Benchmark Pipeline & Unit Test Design

**Date:** 2026-02-28
**Status:** Approved

## Overview

VQC benchmark 프로젝트의 실험 파이프라인을 8개 페이즈로 분리하고, 각 페이즈별 단위 테스트를 서브에이전트가 독립적으로 실행하도록 설계한다.

## Design Decisions

- **Phase scope:** 전체 end-to-end (8 phases)
- **Isolation:** 파일 접근 차단 — 각 테스트 파일은 자체 mock/fixture 포함, 다른 페이즈 참조 불가
- **Output:** 자연어 테스트 리포트 형식 (pass/fail + 설명)
- **Test weight:** 경량 테스트 (maxiter=1-3, 소량 데이터)

## 8 Phases

| Phase | Name | Input | Output | Source |
|-------|------|-------|--------|--------|
| 1 | Data Loading | dataset name | X_train, X_test, y_train, y_test | `src/data_loader.py` |
| 2 | Preprocessing | raw arrays | scaled arrays in [0,π], PCA applied | `src/data_loader.py` |
| 3 | VQC Training | train/test arrays | metrics dict (accuracy, f1, time) | `src/vqc_model.py` |
| 4 | Classical Training | train/test arrays | metrics dict | `src/classical_models.py` |
| 5 | QSVC Training | train/test arrays | metrics dict | `run_experiment7_qsvc.py` |
| 6 | Result Aggregation | per-seed CSVs | summary DataFrame | `run_experiments_final.py` |
| 7 | Figure Generation | CSV results | PNG files | `generate_figures.py` |
| 8 | Paper Build | .tex + figures | PDF | `paper.tex` |

## Test Isolation Strategy

각 테스트 파일은 **self-contained**:
- mock 데이터/fixture를 파일 내부에 직접 생성
- `src/` 모듈만 import (해당 페이즈의 모듈만)
- 다른 페이즈의 테스트 파일이나 결과 파일을 참조하지 않음

```
tests/
  test_phase1_data_loading.py      # numpy로 직접 fixture 생성
  test_phase2_preprocessing.py     # 인라인 mock 데이터
  test_phase3_vqc_training.py      # maxiter=1, 소량 데이터
  test_phase4_classical_training.py
  test_phase5_qsvc_training.py
  test_phase6_aggregation.py       # 인라인 mock CSV 데이터
  test_phase7_figures.py           # mock DataFrame → PNG 생성 확인
  test_phase8_paper_build.py       # .tex 파일 존재 + 빌드 확인
```

## Subagent Execution Model

각 서브에이전트는:
1. 자신의 테스트 파일 **하나만** 실행 (`python -m pytest tests/test_phaseN_xxx.py`)
2. 다른 페이즈의 소스/테스트 파일을 읽지 않음
3. 결과를 자연어 리포트로 출력

## Natural Language Report Format

```
=== Phase 3: VQC Training 테스트 결과 ===

[PASS] VQC 모델 생성: feature map과 ansatz가 올바른 qubit 수로 생성됨
[PASS] VQC 학습 실행: maxiter=1로 학습이 정상 완료됨 (0.8초)
[PASS] 예측 출력 형식: predict() 결과가 학습 레이블과 동일한 클래스 집합
[FAIL] 예측 정확도 범위: 정확도 0.15 — 랜덤 수준(0.33) 이하
       → maxiter=1이므로 낮은 정확도는 예상됨. 단, 0.15는 확인 필요

총 4개 테스트 중 3개 통과, 1개 실패
```

## Key Constraints

- VQC/QSVC 테스트: `maxiter=1~3`, 데이터 10~20개로 제한 (실행 시간 < 30초)
- Figure 테스트: 실제 렌더링 대신 파일 생성 여부만 확인
- Paper build 테스트: xelatex 존재 여부 + 컴파일 성공 여부
