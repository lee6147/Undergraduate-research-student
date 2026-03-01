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
