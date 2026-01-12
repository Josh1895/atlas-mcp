"""Oracle execution for task verification."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path

from atlas.core.task_dag import OracleType, TaskOracle
from atlas.verification.patch_applier import create_patched_checkout
from atlas.verification.test_runner import TestRunner


@dataclass
class OracleResult:
    """Result of running a verification oracle."""

    oracle: TaskOracle
    success: bool
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    duration_seconds: float = 0.0
    error_message: str = ""


class OracleRunner:
    """Runs task oracles on a patched checkout."""

    async def run(
        self,
        repo_path: Path,
        patch: str,
        oracles: list[TaskOracle],
    ) -> list[OracleResult]:
        if not oracles:
            return []

        patched_path, apply_result = create_patched_checkout(repo_path, patch)
        if not patched_path:
            return [
                OracleResult(
                    oracle=oracle,
                    success=False,
                    error_message="Failed to apply patch for oracle execution",
                )
                for oracle in oracles
            ]

        results: list[OracleResult] = []
        try:
            for oracle in oracles:
                result = await self._run_single_oracle(patched_path, oracle)
                results.append(result)
        finally:
            await self._cleanup(patched_path)

        return results

    async def _run_single_oracle(
        self,
        repo_path: Path,
        oracle: TaskOracle,
    ) -> OracleResult:
        start_time = time.time()
        if oracle.oracle_type == OracleType.TEST and not oracle.command:
            runner = TestRunner(timeout_seconds=oracle.timeout_seconds)
            test_result = await runner.run_tests(repo_path)
            return OracleResult(
                oracle=oracle,
                success=test_result.success,
                stdout=test_result.stdout,
                stderr=test_result.stderr,
                exit_code=test_result.exit_code,
                duration_seconds=time.time() - start_time,
                error_message=test_result.execution_error,
            )

        stdout, stderr, exit_code = await self._run_command(
            oracle.command,
            repo_path,
            oracle.timeout_seconds,
        )

        return OracleResult(
            oracle=oracle,
            success=exit_code == 0,
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            duration_seconds=time.time() - start_time,
            error_message="" if exit_code == 0 else stderr[:200],
        )

    async def _run_command(
        self,
        command: str,
        cwd: Path,
        timeout: int,
    ) -> tuple[str, str, int]:
        try:
            import platform
            use_shell = platform.system() == "Windows"

            process = await asyncio.create_subprocess_shell(
                command,
                cwd=str(cwd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                shell=use_shell,
            )

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
            return "", "Oracle execution timed out", -1
        except Exception as exc:
            return "", str(exc), -1

    async def _cleanup(self, patched_path: Path) -> None:
        if patched_path.exists():
            import shutil

            await asyncio.to_thread(shutil.rmtree, patched_path)

