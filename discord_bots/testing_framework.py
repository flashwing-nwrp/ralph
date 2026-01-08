"""
Testing Framework for RALPH Agent Ensemble

Provides comprehensive testing capabilities:
- Unit tests for agent components
- Integration tests for workflows
- Simulation tests for strategies
- Regression tests for model changes
- Performance benchmarks

P2: Important for reliability and quality assurance.
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
import traceback

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("testing_framework")


class TestType(Enum):
    """Types of tests."""
    UNIT = "unit"                    # Unit tests for components
    INTEGRATION = "integration"      # Integration tests
    SIMULATION = "simulation"        # Strategy simulation tests
    REGRESSION = "regression"        # Regression tests
    PERFORMANCE = "performance"      # Performance benchmarks
    SMOKE = "smoke"                  # Quick smoke tests
    E2E = "e2e"                      # End-to-end tests


class TestStatus(Enum):
    """Status of a test."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class TestCase:
    """A single test case."""
    test_id: str
    name: str
    description: str
    test_type: TestType
    category: str  # Agent, strategy, data, etc.

    # Test function
    test_func_name: str

    # Status
    status: TestStatus = TestStatus.PENDING
    run_count: int = 0
    last_run: str = ""
    last_duration_ms: float = 0.0

    # Results
    result_message: str = ""
    error_traceback: str = ""
    assertions_passed: int = 0
    assertions_failed: int = 0

    # Dependencies
    depends_on: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "test_id": self.test_id,
            "name": self.name,
            "description": self.description,
            "test_type": self.test_type.value,
            "category": self.category,
            "test_func_name": self.test_func_name,
            "status": self.status.value,
            "run_count": self.run_count,
            "last_run": self.last_run,
            "last_duration_ms": self.last_duration_ms,
            "result_message": self.result_message,
            "error_traceback": self.error_traceback,
            "assertions_passed": self.assertions_passed,
            "assertions_failed": self.assertions_failed,
            "depends_on": self.depends_on,
            "tags": self.tags
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TestCase":
        return cls(
            test_id=data["test_id"],
            name=data["name"],
            description=data["description"],
            test_type=TestType(data["test_type"]),
            category=data["category"],
            test_func_name=data["test_func_name"],
            status=TestStatus(data.get("status", "pending")),
            run_count=data.get("run_count", 0),
            last_run=data.get("last_run", ""),
            last_duration_ms=data.get("last_duration_ms", 0.0),
            result_message=data.get("result_message", ""),
            error_traceback=data.get("error_traceback", ""),
            assertions_passed=data.get("assertions_passed", 0),
            assertions_failed=data.get("assertions_failed", 0),
            depends_on=data.get("depends_on", []),
            tags=data.get("tags", [])
        )


@dataclass
class TestSuite:
    """A collection of related tests."""
    suite_id: str
    name: str
    description: str
    test_ids: List[str] = field(default_factory=list)

    # Stats
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0

    # Timing
    last_run: str = ""
    total_duration_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "suite_id": self.suite_id,
            "name": self.name,
            "description": self.description,
            "test_ids": self.test_ids,
            "total_tests": self.total_tests,
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "errors": self.errors,
            "last_run": self.last_run,
            "total_duration_ms": self.total_duration_ms
        }


@dataclass
class TestRun:
    """Record of a test run execution."""
    run_id: str
    started_at: str
    completed_at: str = ""
    status: str = "running"

    # Scope
    suite_id: str = ""
    test_ids: List[str] = field(default_factory=list)

    # Results
    total: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    duration_ms: float = 0.0

    # Individual results
    test_results: List[Dict[str, Any]] = field(default_factory=list)


class AssertionError(Exception):
    """Custom assertion error for tests."""
    pass


class TestContext:
    """Context provided to test functions."""

    def __init__(self, test_id: str):
        self.test_id = test_id
        self.assertions_passed = 0
        self.assertions_failed = 0
        self.logs: List[str] = []
        self.data: Dict[str, Any] = {}

    def log(self, message: str):
        """Log a message during test execution."""
        self.logs.append(f"[{datetime.utcnow().isoformat()}] {message}")

    def assert_true(self, condition: bool, message: str = ""):
        """Assert that condition is True."""
        if condition:
            self.assertions_passed += 1
        else:
            self.assertions_failed += 1
            raise AssertionError(f"Expected True, got False. {message}")

    def assert_false(self, condition: bool, message: str = ""):
        """Assert that condition is False."""
        if not condition:
            self.assertions_passed += 1
        else:
            self.assertions_failed += 1
            raise AssertionError(f"Expected False, got True. {message}")

    def assert_equal(self, actual: Any, expected: Any, message: str = ""):
        """Assert that actual equals expected."""
        if actual == expected:
            self.assertions_passed += 1
        else:
            self.assertions_failed += 1
            raise AssertionError(f"Expected {expected}, got {actual}. {message}")

    def assert_not_equal(self, actual: Any, expected: Any, message: str = ""):
        """Assert that actual does not equal expected."""
        if actual != expected:
            self.assertions_passed += 1
        else:
            self.assertions_failed += 1
            raise AssertionError(f"Expected not {expected}, but got it. {message}")

    def assert_greater(self, actual: float, expected: float, message: str = ""):
        """Assert that actual > expected."""
        if actual > expected:
            self.assertions_passed += 1
        else:
            self.assertions_failed += 1
            raise AssertionError(f"Expected {actual} > {expected}. {message}")

    def assert_less(self, actual: float, expected: float, message: str = ""):
        """Assert that actual < expected."""
        if actual < expected:
            self.assertions_passed += 1
        else:
            self.assertions_failed += 1
            raise AssertionError(f"Expected {actual} < {expected}. {message}")

    def assert_in_range(self, value: float, min_val: float, max_val: float, message: str = ""):
        """Assert that value is within range."""
        if min_val <= value <= max_val:
            self.assertions_passed += 1
        else:
            self.assertions_failed += 1
            raise AssertionError(f"Expected {value} in range [{min_val}, {max_val}]. {message}")

    def assert_not_none(self, value: Any, message: str = ""):
        """Assert that value is not None."""
        if value is not None:
            self.assertions_passed += 1
        else:
            self.assertions_failed += 1
            raise AssertionError(f"Expected not None. {message}")

    def assert_raises(self, exception_type: type, func: Callable, *args, **kwargs):
        """Assert that function raises expected exception."""
        try:
            func(*args, **kwargs)
            self.assertions_failed += 1
            raise AssertionError(f"Expected {exception_type.__name__} to be raised")
        except exception_type:
            self.assertions_passed += 1
        except Exception as e:
            self.assertions_failed += 1
            raise AssertionError(f"Expected {exception_type.__name__}, got {type(e).__name__}")


class TestingFramework:
    """
    Central testing framework for RALPH.

    Provides:
    - Test registration and discovery
    - Test execution and reporting
    - Suite management
    - Coverage tracking
    """

    def __init__(self, project_dir: str = None):
        self.project_dir = Path(project_dir or os.getenv("RALPH_PROJECT_DIR", "."))
        self.tests_file = self.project_dir / "test_registry.json"
        self.results_file = self.project_dir / "test_results.json"

        # Test registry
        self.tests: Dict[str, TestCase] = {}
        self.suites: Dict[str, TestSuite] = {}
        self.test_counter = 0
        self.suite_counter = 0
        self.run_counter = 0

        # Registered test functions
        self.test_functions: Dict[str, Callable] = {}

        # Results
        self.test_runs: List[TestRun] = []

        self._lock = asyncio.Lock()
        self._load_tests()

    def _load_tests(self):
        """Load test registry from file."""
        try:
            if self.tests_file.exists():
                with open(self.tests_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.test_counter = data.get("test_counter", 0)
                    self.suite_counter = data.get("suite_counter", 0)

                    for test_data in data.get("tests", []):
                        test = TestCase.from_dict(test_data)
                        self.tests[test.test_id] = test

                    for suite_data in data.get("suites", []):
                        suite = TestSuite(**suite_data)
                        self.suites[suite.suite_id] = suite

                    logger.info(f"Loaded {len(self.tests)} tests in {len(self.suites)} suites")

        except Exception as e:
            logger.error(f"Failed to load test registry: {e}")

    async def _save_tests(self):
        """Save test registry to file."""
        async with self._lock:
            try:
                data = {
                    "test_counter": self.test_counter,
                    "suite_counter": self.suite_counter,
                    "tests": [t.to_dict() for t in self.tests.values()],
                    "suites": [s.to_dict() for s in self.suites.values()]
                }

                with open(self.tests_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)

            except Exception as e:
                logger.error(f"Failed to save test registry: {e}")

    # =========================================================================
    # TEST REGISTRATION
    # =========================================================================

    def register_test(
        self,
        name: str,
        description: str,
        test_type: TestType,
        category: str,
        test_func: Callable,
        tags: List[str] = None,
        depends_on: List[str] = None
    ) -> TestCase:
        """
        Register a new test.

        Args:
            name: Test name
            description: What the test verifies
            test_type: Type of test
            category: Category (agent, strategy, data, etc.)
            test_func: Test function (takes TestContext as first arg)
            tags: Tags for filtering
            depends_on: Test IDs this depends on

        Returns:
            Created test case
        """
        self.test_counter += 1
        test_id = f"TEST-{self.test_counter:04d}"

        # Register function
        func_name = f"{test_id}_{name.replace(' ', '_').lower()}"
        self.test_functions[func_name] = test_func

        test = TestCase(
            test_id=test_id,
            name=name,
            description=description,
            test_type=test_type,
            category=category,
            test_func_name=func_name,
            tags=tags or [],
            depends_on=depends_on or []
        )

        self.tests[test_id] = test
        logger.info(f"Registered test: {test_id} - {name}")

        return test

    def create_suite(
        self,
        name: str,
        description: str,
        test_ids: List[str] = None
    ) -> TestSuite:
        """Create a test suite."""
        self.suite_counter += 1
        suite_id = f"SUITE-{self.suite_counter:03d}"

        suite = TestSuite(
            suite_id=suite_id,
            name=name,
            description=description,
            test_ids=test_ids or [],
            total_tests=len(test_ids) if test_ids else 0
        )

        self.suites[suite_id] = suite
        return suite

    def add_to_suite(self, suite_id: str, test_id: str) -> bool:
        """Add a test to a suite."""
        if suite_id not in self.suites or test_id not in self.tests:
            return False

        suite = self.suites[suite_id]
        if test_id not in suite.test_ids:
            suite.test_ids.append(test_id)
            suite.total_tests = len(suite.test_ids)

        return True

    # =========================================================================
    # TEST EXECUTION
    # =========================================================================

    async def run_test(self, test_id: str) -> TestCase:
        """
        Run a single test.

        Args:
            test_id: ID of test to run

        Returns:
            Updated test case with results
        """
        if test_id not in self.tests:
            raise ValueError(f"Test {test_id} not found")

        test = self.tests[test_id]
        test.status = TestStatus.RUNNING
        test.run_count += 1
        test.last_run = datetime.utcnow().isoformat()

        # Check dependencies
        for dep_id in test.depends_on:
            if dep_id in self.tests:
                dep = self.tests[dep_id]
                if dep.status not in [TestStatus.PASSED]:
                    test.status = TestStatus.SKIPPED
                    test.result_message = f"Skipped: dependency {dep_id} not passed"
                    await self._save_tests()
                    return test

        # Get test function
        if test.test_func_name not in self.test_functions:
            test.status = TestStatus.ERROR
            test.result_message = f"Test function {test.test_func_name} not registered"
            await self._save_tests()
            return test

        test_func = self.test_functions[test.test_func_name]
        ctx = TestContext(test_id)

        start_time = time.perf_counter()

        try:
            # Run test
            if asyncio.iscoroutinefunction(test_func):
                await test_func(ctx)
            else:
                test_func(ctx)

            # Check assertions
            if ctx.assertions_failed > 0:
                test.status = TestStatus.FAILED
                test.result_message = f"Assertions: {ctx.assertions_passed} passed, {ctx.assertions_failed} failed"
            else:
                test.status = TestStatus.PASSED
                test.result_message = f"All {ctx.assertions_passed} assertions passed"

        except AssertionError as e:
            test.status = TestStatus.FAILED
            test.result_message = str(e)
            test.error_traceback = traceback.format_exc()

        except Exception as e:
            test.status = TestStatus.ERROR
            test.result_message = f"Error: {str(e)}"
            test.error_traceback = traceback.format_exc()

        test.last_duration_ms = (time.perf_counter() - start_time) * 1000
        test.assertions_passed = ctx.assertions_passed
        test.assertions_failed = ctx.assertions_failed

        await self._save_tests()
        logger.info(f"Test {test_id}: {test.status.value} ({test.last_duration_ms:.1f}ms)")

        return test

    async def run_suite(self, suite_id: str) -> TestRun:
        """
        Run all tests in a suite.

        Args:
            suite_id: Suite to run

        Returns:
            Test run results
        """
        if suite_id not in self.suites:
            raise ValueError(f"Suite {suite_id} not found")

        suite = self.suites[suite_id]

        self.run_counter += 1
        run = TestRun(
            run_id=f"RUN-{self.run_counter:05d}",
            started_at=datetime.utcnow().isoformat(),
            suite_id=suite_id,
            test_ids=suite.test_ids.copy(),
            total=len(suite.test_ids)
        )

        start_time = time.perf_counter()

        # Run tests in order
        for test_id in suite.test_ids:
            result = await self.run_test(test_id)

            run.test_results.append({
                "test_id": test_id,
                "status": result.status.value,
                "duration_ms": result.last_duration_ms,
                "message": result.result_message
            })

            if result.status == TestStatus.PASSED:
                run.passed += 1
            elif result.status == TestStatus.FAILED:
                run.failed += 1
            elif result.status == TestStatus.SKIPPED:
                run.skipped += 1
            elif result.status == TestStatus.ERROR:
                run.errors += 1

        run.completed_at = datetime.utcnow().isoformat()
        run.duration_ms = (time.perf_counter() - start_time) * 1000
        run.status = "passed" if run.failed == 0 and run.errors == 0 else "failed"

        # Update suite stats
        suite.passed = run.passed
        suite.failed = run.failed
        suite.skipped = run.skipped
        suite.errors = run.errors
        suite.last_run = run.started_at
        suite.total_duration_ms = run.duration_ms

        self.test_runs.append(run)
        await self._save_tests()

        return run

    async def run_by_type(self, test_type: TestType) -> TestRun:
        """Run all tests of a specific type."""
        test_ids = [t.test_id for t in self.tests.values() if t.test_type == test_type]

        self.run_counter += 1
        run = TestRun(
            run_id=f"RUN-{self.run_counter:05d}",
            started_at=datetime.utcnow().isoformat(),
            test_ids=test_ids,
            total=len(test_ids)
        )

        start_time = time.perf_counter()

        for test_id in test_ids:
            result = await self.run_test(test_id)

            run.test_results.append({
                "test_id": test_id,
                "status": result.status.value,
                "duration_ms": result.last_duration_ms
            })

            if result.status == TestStatus.PASSED:
                run.passed += 1
            elif result.status == TestStatus.FAILED:
                run.failed += 1
            elif result.status == TestStatus.SKIPPED:
                run.skipped += 1
            else:
                run.errors += 1

        run.completed_at = datetime.utcnow().isoformat()
        run.duration_ms = (time.perf_counter() - start_time) * 1000
        run.status = "passed" if run.failed == 0 and run.errors == 0 else "failed"

        self.test_runs.append(run)
        return run

    async def run_by_tag(self, tag: str) -> TestRun:
        """Run all tests with a specific tag."""
        test_ids = [t.test_id for t in self.tests.values() if tag in t.tags]

        self.run_counter += 1
        run = TestRun(
            run_id=f"RUN-{self.run_counter:05d}",
            started_at=datetime.utcnow().isoformat(),
            test_ids=test_ids,
            total=len(test_ids)
        )

        start_time = time.perf_counter()

        for test_id in test_ids:
            result = await self.run_test(test_id)
            run.test_results.append({
                "test_id": test_id,
                "status": result.status.value,
                "duration_ms": result.last_duration_ms
            })

            if result.status == TestStatus.PASSED:
                run.passed += 1
            elif result.status == TestStatus.FAILED:
                run.failed += 1
            elif result.status == TestStatus.SKIPPED:
                run.skipped += 1
            else:
                run.errors += 1

        run.completed_at = datetime.utcnow().isoformat()
        run.duration_ms = (time.perf_counter() - start_time) * 1000
        run.status = "passed" if run.failed == 0 and run.errors == 0 else "failed"

        self.test_runs.append(run)
        return run

    async def run_all(self) -> TestRun:
        """Run all registered tests."""
        test_ids = list(self.tests.keys())

        self.run_counter += 1
        run = TestRun(
            run_id=f"RUN-{self.run_counter:05d}",
            started_at=datetime.utcnow().isoformat(),
            test_ids=test_ids,
            total=len(test_ids)
        )

        start_time = time.perf_counter()

        for test_id in test_ids:
            result = await self.run_test(test_id)
            run.test_results.append({
                "test_id": test_id,
                "status": result.status.value,
                "duration_ms": result.last_duration_ms
            })

            if result.status == TestStatus.PASSED:
                run.passed += 1
            elif result.status == TestStatus.FAILED:
                run.failed += 1
            elif result.status == TestStatus.SKIPPED:
                run.skipped += 1
            else:
                run.errors += 1

        run.completed_at = datetime.utcnow().isoformat()
        run.duration_ms = (time.perf_counter() - start_time) * 1000
        run.status = "passed" if run.failed == 0 and run.errors == 0 else "failed"

        self.test_runs.append(run)
        return run

    # =========================================================================
    # REPORTING
    # =========================================================================

    def get_test_summary(self) -> str:
        """Get test summary for Discord."""
        if not self.tests:
            return "No tests registered."

        output = ["## 🧪 Test Summary\n"]

        # By status
        passed = [t for t in self.tests.values() if t.status == TestStatus.PASSED]
        failed = [t for t in self.tests.values() if t.status == TestStatus.FAILED]
        pending = [t for t in self.tests.values() if t.status == TestStatus.PENDING]

        output.append(f"**Total:** {len(self.tests)} tests")
        output.append(f"  ✅ Passed: {len(passed)}")
        output.append(f"  ❌ Failed: {len(failed)}")
        output.append(f"  ⏳ Pending: {len(pending)}")

        # By type
        output.append("\n**By Type:**")
        by_type: Dict[TestType, List[TestCase]] = {}
        for test in self.tests.values():
            if test.test_type not in by_type:
                by_type[test.test_type] = []
            by_type[test.test_type].append(test)

        for test_type, tests in sorted(by_type.items(), key=lambda x: x[0].value):
            passed_count = len([t for t in tests if t.status == TestStatus.PASSED])
            output.append(f"  {test_type.value}: {passed_count}/{len(tests)} passed")

        # Suites
        if self.suites:
            output.append(f"\n**Suites:** {len(self.suites)}")
            for suite in self.suites.values():
                status = "✅" if suite.failed == 0 and suite.errors == 0 else "❌"
                output.append(f"  {status} {suite.name}: {suite.passed}/{suite.total_tests}")

        return "\n".join(output)

    def get_test_details(self, test_id: str) -> str:
        """Get detailed test information."""
        if test_id not in self.tests:
            return f"Test {test_id} not found."

        test = self.tests[test_id]

        status_emoji = {
            TestStatus.PASSED: "✅",
            TestStatus.FAILED: "❌",
            TestStatus.PENDING: "⏳",
            TestStatus.RUNNING: "🔄",
            TestStatus.SKIPPED: "⏭️",
            TestStatus.ERROR: "💥"
        }

        output = [
            f"## Test: {test.name}\n",
            f"**ID:** {test.test_id}",
            f"**Status:** {status_emoji.get(test.status, '❓')} {test.status.value}",
            f"**Type:** {test.test_type.value}",
            f"**Category:** {test.category}",
            f"",
            f"_{test.description}_",
            f"",
            f"**Runs:** {test.run_count}",
            f"**Last Run:** {test.last_run or 'Never'}",
            f"**Duration:** {test.last_duration_ms:.1f}ms",
            f"**Assertions:** {test.assertions_passed} passed, {test.assertions_failed} failed"
        ]

        if test.result_message:
            output.append(f"\n**Result:** {test.result_message}")

        if test.error_traceback:
            output.append(f"\n**Error:**\n```\n{test.error_traceback[:500]}\n```")

        if test.tags:
            output.append(f"\n**Tags:** {', '.join(test.tags)}")

        if test.depends_on:
            output.append(f"**Depends On:** {', '.join(test.depends_on)}")

        return "\n".join(output)

    def get_run_results(self, run_id: str = None) -> str:
        """Get test run results."""
        if run_id:
            run = next((r for r in self.test_runs if r.run_id == run_id), None)
            if not run:
                return f"Run {run_id} not found."
            runs = [run]
        else:
            runs = self.test_runs[-5:]  # Last 5 runs

        if not runs:
            return "No test runs recorded."

        output = ["## Test Run Results\n"]

        for run in reversed(runs):
            status = "✅" if run.status == "passed" else "❌"
            output.append(f"{status} **{run.run_id}**")
            output.append(f"   {run.started_at[:16]} | {run.duration_ms:.1f}ms")
            output.append(
                f"   Passed: {run.passed} | Failed: {run.failed} | "
                f"Skipped: {run.skipped} | Errors: {run.errors}"
            )

            if run.failed > 0 or run.errors > 0:
                failed_tests = [
                    r for r in run.test_results
                    if r["status"] in ["failed", "error"]
                ]
                for result in failed_tests[:3]:
                    output.append(f"   ❌ {result['test_id']}: {result.get('message', '')[:50]}")

            output.append("")

        return "\n".join(output)


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_testing_framework: Optional[TestingFramework] = None


def get_testing_framework() -> TestingFramework:
    """Get or create the testing framework instance."""
    global _testing_framework
    if _testing_framework is None:
        _testing_framework = TestingFramework()
    return _testing_framework


def set_testing_framework(framework: TestingFramework):
    """Set the testing framework instance."""
    global _testing_framework
    _testing_framework = framework
