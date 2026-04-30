from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Callable, Sequence

from pydantic import BaseModel, ConfigDict


Runner = Callable[[Sequence[str], Path], subprocess.CompletedProcess]


class TestResult(BaseModel):
    __test__ = False
    model_config = ConfigDict(frozen=True)

    passed: int = 0
    failed: int = 0
    errors: int = 0
    coverage_pct: float | None = None
    failed_tests: tuple[str, ...] = ()
    success: bool


class TestRunner:
    __test__ = False

    def __init__(self, *, runner: Runner | None = None) -> None:
        self._runner = runner or self._run

    def run(self, working_dir: Path = Path(".")) -> TestResult:
        report_path = working_dir / ".test_report.json"
        command = [
            "python",
            "-m",
            "pytest",
            "tests/",
            "--tb=short",
            "--json-report",
            "--json-report-file=.test_report.json",
        ]
        completed = self._runner(command, working_dir)
        if report_path.exists():
            return self._from_report(report_path, completed.returncode == 0)
        return TestResult(success=False, failed=1, failed_tests=("pytest-json-report missing",))

    def _from_report(self, report_path: Path, success: bool) -> TestResult:
        data = json.loads(report_path.read_text(encoding="utf-8"))
        summary = data.get("summary", {})
        failed_tests = []
        for test in data.get("tests", []):
            if test.get("outcome") in {"failed", "error"}:
                failed_tests.append(test.get("nodeid", "unknown"))
        return TestResult(
            passed=int(summary.get("passed", 0)),
            failed=int(summary.get("failed", 0)),
            errors=int(summary.get("errors", 0)),
            coverage_pct=None,
            failed_tests=tuple(failed_tests),
            success=success,
        )

    def _run(self, command: Sequence[str], cwd: Path) -> subprocess.CompletedProcess:
        return subprocess.run(command, cwd=cwd, text=True, capture_output=True)
