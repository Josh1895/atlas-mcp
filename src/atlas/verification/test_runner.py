"""Test runner with auto-detection for ATLAS.

This module handles running tests on patched code to verify correctness.
It can auto-detect the test framework and command from repository files.
"""

import asyncio
import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class TestFramework(str, Enum):
    """Supported test frameworks."""

    PYTEST = "pytest"
    UNITTEST = "unittest"
    NOSE = "nose"
    NPM = "npm"
    JEST = "jest"
    MOCHA = "mocha"
    VITEST = "vitest"
    CARGO = "cargo"
    GO = "go"
    MAVEN = "maven"
    GRADLE = "gradle"
    RSPEC = "rspec"
    MINITEST = "minitest"
    PHPUNIT = "phpunit"
    DOTNET = "dotnet"
    UNKNOWN = "unknown"


@dataclass
class TestCase:
    """A single test case result."""

    name: str
    passed: bool
    duration_ms: float = 0.0
    error_message: str = ""
    stack_trace: str = ""
    file_path: str = ""
    line_number: int = 0


@dataclass
class TestResult:
    """Result from running tests."""

    success: bool  # All tests passed
    framework: TestFramework = TestFramework.UNKNOWN
    command_used: str = ""
    exit_code: int = 0

    # Test counts
    total: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0

    # Detailed results
    test_cases: List[TestCase] = field(default_factory=list)
    failed_tests: List[TestCase] = field(default_factory=list)

    # Output
    stdout: str = ""
    stderr: str = ""
    duration_seconds: float = 0.0

    # Errors during execution
    execution_error: str = ""

    @property
    def outcome_vector(self) -> Tuple[int, int, int, int]:
        """Return a tuple representing test outcome for clustering."""
        return (self.passed, self.failed, self.skipped, self.errors)

    @property
    def failure_signature(self) -> str:
        """Generate a signature of failures for clustering."""
        if not self.failed_tests:
            return "all_pass"
        # Sort failed test names for consistent signature
        failed_names = sorted([t.name for t in self.failed_tests])
        return "|".join(failed_names)


@dataclass
class DetectedTestConfig:
    """Detected test configuration for a repository."""

    framework: TestFramework
    command: str
    working_directory: str = "."
    confidence: float = 1.0
    setup_command: Optional[str] = None  # e.g., "npm install"


class TestFrameworkDetector:
    """Detects test framework and command from repository structure."""

    def detect(self, repo_path: Path) -> List[DetectedTestConfig]:
        """Detect test framework(s) from repository.

        Args:
            repo_path: Path to the repository

        Returns:
            List of detected test configurations, ordered by confidence
        """
        configs = []

        # Python detection
        python_config = self._detect_python(repo_path)
        if python_config:
            configs.append(python_config)

        # Node.js detection
        node_config = self._detect_node(repo_path)
        if node_config:
            configs.append(node_config)

        # Rust detection
        rust_config = self._detect_rust(repo_path)
        if rust_config:
            configs.append(rust_config)

        # Go detection
        go_config = self._detect_go(repo_path)
        if go_config:
            configs.append(go_config)

        # Java detection (Maven/Gradle)
        java_config = self._detect_java(repo_path)
        if java_config:
            configs.append(java_config)

        # Ruby detection
        ruby_config = self._detect_ruby(repo_path)
        if ruby_config:
            configs.append(ruby_config)

        # .NET detection
        dotnet_config = self._detect_dotnet(repo_path)
        if dotnet_config:
            configs.append(dotnet_config)

        # Sort by confidence
        configs.sort(key=lambda x: x.confidence, reverse=True)

        return configs

    def _detect_python(self, repo_path: Path) -> Optional[DetectedTestConfig]:
        """Detect Python test framework."""
        # Check for pyproject.toml
        pyproject = repo_path / "pyproject.toml"
        if pyproject.exists():
            content = pyproject.read_text()
            if "pytest" in content:
                return DetectedTestConfig(
                    framework=TestFramework.PYTEST,
                    command="pytest",
                    confidence=1.0,
                )

        # Check for pytest.ini
        if (repo_path / "pytest.ini").exists():
            return DetectedTestConfig(
                framework=TestFramework.PYTEST,
                command="pytest",
                confidence=1.0,
            )

        # Check for setup.py/setup.cfg
        if (repo_path / "setup.py").exists() or (repo_path / "setup.cfg").exists():
            return DetectedTestConfig(
                framework=TestFramework.PYTEST,
                command="pytest",
                confidence=0.8,
            )

        # Check for test files
        test_files = list(repo_path.rglob("test_*.py")) + list(repo_path.rglob("*_test.py"))
        if test_files:
            return DetectedTestConfig(
                framework=TestFramework.PYTEST,
                command="pytest",
                confidence=0.7,
            )

        # Check for tests directory
        if (repo_path / "tests").exists():
            return DetectedTestConfig(
                framework=TestFramework.PYTEST,
                command="pytest tests/",
                confidence=0.6,
            )

        return None

    def _detect_node(self, repo_path: Path) -> Optional[DetectedTestConfig]:
        """Detect Node.js test framework."""
        package_json = repo_path / "package.json"
        if not package_json.exists():
            return None

        try:
            content = json.loads(package_json.read_text())
        except json.JSONDecodeError:
            return None

        scripts = content.get("scripts", {})
        dev_deps = content.get("devDependencies", {})
        deps = content.get("dependencies", {})
        all_deps = {**deps, **dev_deps}

        # Check for test script
        test_script = scripts.get("test", "")

        # Detect specific frameworks
        if "vitest" in all_deps or "vitest" in test_script:
            return DetectedTestConfig(
                framework=TestFramework.VITEST,
                command="npm test" if test_script else "npx vitest",
                setup_command="npm install",
                confidence=1.0,
            )

        if "jest" in all_deps or "jest" in test_script:
            return DetectedTestConfig(
                framework=TestFramework.JEST,
                command="npm test" if test_script else "npx jest",
                setup_command="npm install",
                confidence=1.0,
            )

        if "mocha" in all_deps or "mocha" in test_script:
            return DetectedTestConfig(
                framework=TestFramework.MOCHA,
                command="npm test" if test_script else "npx mocha",
                setup_command="npm install",
                confidence=1.0,
            )

        # Generic npm test
        if test_script and test_script != 'echo "Error: no test specified" && exit 1':
            return DetectedTestConfig(
                framework=TestFramework.NPM,
                command="npm test",
                setup_command="npm install",
                confidence=0.8,
            )

        return None

    def _detect_rust(self, repo_path: Path) -> Optional[DetectedTestConfig]:
        """Detect Rust test framework."""
        if (repo_path / "Cargo.toml").exists():
            return DetectedTestConfig(
                framework=TestFramework.CARGO,
                command="cargo test",
                confidence=1.0,
            )
        return None

    def _detect_go(self, repo_path: Path) -> Optional[DetectedTestConfig]:
        """Detect Go test framework."""
        if (repo_path / "go.mod").exists():
            return DetectedTestConfig(
                framework=TestFramework.GO,
                command="go test ./...",
                confidence=1.0,
            )

        # Check for go files with tests
        test_files = list(repo_path.rglob("*_test.go"))
        if test_files:
            return DetectedTestConfig(
                framework=TestFramework.GO,
                command="go test ./...",
                confidence=0.8,
            )

        return None

    def _detect_java(self, repo_path: Path) -> Optional[DetectedTestConfig]:
        """Detect Java test framework (Maven/Gradle)."""
        if (repo_path / "pom.xml").exists():
            return DetectedTestConfig(
                framework=TestFramework.MAVEN,
                command="mvn test",
                confidence=1.0,
            )

        if (repo_path / "build.gradle").exists() or (repo_path / "build.gradle.kts").exists():
            return DetectedTestConfig(
                framework=TestFramework.GRADLE,
                command="./gradlew test" if (repo_path / "gradlew").exists() else "gradle test",
                confidence=1.0,
            )

        return None

    def _detect_ruby(self, repo_path: Path) -> Optional[DetectedTestConfig]:
        """Detect Ruby test framework."""
        gemfile = repo_path / "Gemfile"
        if gemfile.exists():
            content = gemfile.read_text()
            if "rspec" in content:
                return DetectedTestConfig(
                    framework=TestFramework.RSPEC,
                    command="bundle exec rspec",
                    setup_command="bundle install",
                    confidence=1.0,
                )
            if "minitest" in content:
                return DetectedTestConfig(
                    framework=TestFramework.MINITEST,
                    command="bundle exec rake test",
                    setup_command="bundle install",
                    confidence=1.0,
                )

        # Check for spec directory (RSpec)
        if (repo_path / "spec").exists():
            return DetectedTestConfig(
                framework=TestFramework.RSPEC,
                command="rspec",
                confidence=0.7,
            )

        return None

    def _detect_dotnet(self, repo_path: Path) -> Optional[DetectedTestConfig]:
        """Detect .NET test framework."""
        csproj_files = list(repo_path.rglob("*.csproj"))
        if csproj_files:
            return DetectedTestConfig(
                framework=TestFramework.DOTNET,
                command="dotnet test",
                confidence=1.0,
            )

        if (repo_path / "*.sln").exists():
            return DetectedTestConfig(
                framework=TestFramework.DOTNET,
                command="dotnet test",
                confidence=1.0,
            )

        return None


class TestRunner:
    """Runs tests and parses results."""

    def __init__(
        self,
        timeout_seconds: int = 300,
        capture_output: bool = True,
    ):
        self.timeout_seconds = timeout_seconds
        self.capture_output = capture_output
        self.detector = TestFrameworkDetector()

    async def run_tests(
        self,
        repo_path: Path,
        test_command: Optional[str] = None,
        run_setup: bool = True,
    ) -> TestResult:
        """Run tests in a repository.

        Args:
            repo_path: Path to the repository
            test_command: Optional explicit test command (auto-detects if None)
            run_setup: Whether to run setup commands (e.g., npm install)

        Returns:
            TestResult with test outcomes
        """
        result = TestResult(success=False)

        # Determine test command
        if test_command:
            command = test_command
            result.framework = TestFramework.UNKNOWN
            result.command_used = command
        else:
            configs = self.detector.detect(repo_path)
            if not configs:
                result.execution_error = "Could not detect test framework"
                return result

            config = configs[0]
            result.framework = config.framework
            command = config.command
            result.command_used = command

            # Run setup if needed
            if run_setup and config.setup_command:
                setup_result = await self._run_command(
                    config.setup_command,
                    repo_path,
                    timeout=120,
                )
                if setup_result[2] != 0:  # exit code
                    result.execution_error = f"Setup failed: {setup_result[1]}"
                    return result

        # Run tests
        import time
        start_time = time.time()

        stdout, stderr, exit_code = await self._run_command(
            command,
            repo_path,
            timeout=self.timeout_seconds,
        )

        result.duration_seconds = time.time() - start_time
        result.stdout = stdout
        result.stderr = stderr
        result.exit_code = exit_code

        # Parse results based on framework
        self._parse_results(result)

        return result

    async def _run_command(
        self,
        command: str,
        cwd: Path,
        timeout: int = 300,
    ) -> Tuple[str, str, int]:
        """Run a shell command.

        Args:
            command: Command to run
            cwd: Working directory
            timeout: Timeout in seconds

        Returns:
            Tuple of (stdout, stderr, exit_code)
        """
        try:
            # Use shell on Windows, direct execution on Unix
            import platform
            use_shell = platform.system() == "Windows"

            process = await asyncio.create_subprocess_shell(
                command,
                cwd=str(cwd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                shell=use_shell,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout,
                )
                return (
                    stdout.decode("utf-8", errors="replace"),
                    stderr.decode("utf-8", errors="replace"),
                    process.returncode or 0,
                )
            except asyncio.TimeoutError:
                process.kill()
                return "", "Test execution timed out", -1

        except Exception as e:
            return "", str(e), -1

    def _parse_results(self, result: TestResult) -> None:
        """Parse test output to extract results.

        Args:
            result: TestResult to update with parsed data
        """
        output = result.stdout + result.stderr

        if result.framework == TestFramework.PYTEST:
            self._parse_pytest(result, output)
        elif result.framework in (TestFramework.JEST, TestFramework.VITEST):
            self._parse_jest(result, output)
        elif result.framework == TestFramework.CARGO:
            self._parse_cargo(result, output)
        elif result.framework == TestFramework.GO:
            self._parse_go(result, output)
        else:
            # Generic parsing based on exit code
            result.success = result.exit_code == 0

    def _parse_pytest(self, result: TestResult, output: str) -> None:
        """Parse pytest output."""
        # Look for summary line: "X passed, Y failed, Z skipped"
        summary_match = re.search(
            r'(\d+) passed(?:.*?(\d+) failed)?(?:.*?(\d+) skipped)?(?:.*?(\d+) error)?',
            output,
        )
        if summary_match:
            result.passed = int(summary_match.group(1) or 0)
            result.failed = int(summary_match.group(2) or 0)
            result.skipped = int(summary_match.group(3) or 0)
            result.errors = int(summary_match.group(4) or 0)
            result.total = result.passed + result.failed + result.skipped + result.errors

        # Extract failed tests
        failed_pattern = re.compile(r'FAILED (.+?)::([\w_]+)')
        for match in failed_pattern.finditer(output):
            test_case = TestCase(
                name=f"{match.group(1)}::{match.group(2)}",
                passed=False,
                file_path=match.group(1),
            )
            result.failed_tests.append(test_case)
            result.test_cases.append(test_case)

        result.success = result.exit_code == 0 and result.failed == 0 and result.errors == 0

    def _parse_jest(self, result: TestResult, output: str) -> None:
        """Parse Jest/Vitest output."""
        # Look for "Tests: X passed, Y failed, Z total"
        tests_match = re.search(
            r'Tests:\s*(?:(\d+) passed)?(?:,?\s*(\d+) failed)?(?:,?\s*(\d+) skipped)?(?:,?\s*(\d+) total)?',
            output,
        )
        if tests_match:
            result.passed = int(tests_match.group(1) or 0)
            result.failed = int(tests_match.group(2) or 0)
            result.skipped = int(tests_match.group(3) or 0)
            result.total = int(tests_match.group(4) or result.passed + result.failed + result.skipped)

        # Extract failed tests
        failed_pattern = re.compile(r'✕ (.+)')
        for match in failed_pattern.finditer(output):
            test_case = TestCase(name=match.group(1), passed=False)
            result.failed_tests.append(test_case)
            result.test_cases.append(test_case)

        result.success = result.exit_code == 0 and result.failed == 0

    def _parse_cargo(self, result: TestResult, output: str) -> None:
        """Parse Cargo test output."""
        # Look for "test result: ok. X passed; Y failed; Z ignored"
        summary_match = re.search(
            r'test result: (ok|FAILED)\.\s*(\d+) passed;\s*(\d+) failed;\s*(\d+) ignored',
            output,
        )
        if summary_match:
            result.passed = int(summary_match.group(2))
            result.failed = int(summary_match.group(3))
            result.skipped = int(summary_match.group(4))
            result.total = result.passed + result.failed + result.skipped

        # Extract failed tests
        failed_pattern = re.compile(r'test ([\w:]+) \.\.\. FAILED')
        for match in failed_pattern.finditer(output):
            test_case = TestCase(name=match.group(1), passed=False)
            result.failed_tests.append(test_case)
            result.test_cases.append(test_case)

        result.success = result.exit_code == 0 and result.failed == 0

    def _parse_go(self, result: TestResult, output: str) -> None:
        """Parse Go test output."""
        # Count PASS and FAIL
        result.passed = len(re.findall(r'^--- PASS:', output, re.MULTILINE))
        result.failed = len(re.findall(r'^--- FAIL:', output, re.MULTILINE))
        result.skipped = len(re.findall(r'^--- SKIP:', output, re.MULTILINE))
        result.total = result.passed + result.failed + result.skipped

        # Extract failed tests
        failed_pattern = re.compile(r'--- FAIL: ([\w/]+)')
        for match in failed_pattern.finditer(output):
            test_case = TestCase(name=match.group(1), passed=False)
            result.failed_tests.append(test_case)
            result.test_cases.append(test_case)

        result.success = result.exit_code == 0 and result.failed == 0


async def run_tests_on_patch(
    repo_path: Path,
    patch_content: str,
    test_command: Optional[str] = None,
    timeout_seconds: int = 300,
) -> Tuple[TestResult, Optional[Path]]:
    """Apply a patch and run tests on the result.

    Args:
        repo_path: Path to the original repository
        patch_content: Unified diff to apply
        test_command: Optional test command (auto-detects if None)
        timeout_seconds: Test timeout

    Returns:
        Tuple of (TestResult, patched_repo_path or None on failure)
    """
    from atlas.verification.patch_applier import create_patched_checkout

    # Create patched checkout
    patched_path, apply_result = create_patched_checkout(repo_path, patch_content)

    if not patched_path:
        return TestResult(
            success=False,
            execution_error=f"Failed to apply patch: {'; '.join(apply_result.errors)}",
        ), None

    # Run tests
    runner = TestRunner(timeout_seconds=timeout_seconds)
    test_result = await runner.run_tests(patched_path, test_command)

    return test_result, patched_path


async def cleanup_patched_checkout(patched_path: Path) -> None:
    """Clean up a patched checkout directory.

    Args:
        patched_path: Path to clean up
    """
    if patched_path and patched_path.exists():
        shutil.rmtree(patched_path)
