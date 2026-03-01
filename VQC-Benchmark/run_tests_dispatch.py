"""
Subagent Test Dispatch: 각 페이즈 테스트를 독립적으로 실행하고 자연어 리포트를 생성한다.
사용법: python run_tests_dispatch.py [phase_number]
       python run_tests_dispatch.py         # 전체 실행
       python run_tests_dispatch.py 3       # Phase 3만 실행
"""
import subprocess
import sys
import os

sys.stdout.reconfigure(encoding="utf-8")

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
        timeout=180,
    )
    stdout = result.stdout.decode("utf-8", errors="replace") if result.stdout else ""
    stderr = result.stderr.decode("utf-8", errors="replace") if result.stderr else ""

    return name, result.returncode, stdout + stderr


def format_phase_report(name, returncode, output):
    """페이즈 결과를 자연어로 포맷한다."""
    lines = []
    lines.append(f"=== {name} ===")

    if returncode == "SKIP":
        lines.append(f"[SKIP] {output}")
        return "\n".join(lines)

    # Parse pytest verbose output for test results
    for line in output.split("\n"):
        if "PASSED" in line and "::" in line:
            test_name = line.split("::")[1].split(" ")[0] if "::" in line else line
            lines.append(f"[PASS] {test_name}")
        elif "FAILED" in line and "::" in line:
            test_name = line.split("::")[1].split(" ")[0] if "::" in line else line
            lines.append(f"[FAIL] {test_name}")
        elif "SKIPPED" in line and "::" in line:
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
        print("VQC Benchmark Pipeline - 전체 단위 테스트 리포트")
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
