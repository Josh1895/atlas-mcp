"""Tests for RepoScout."""

import pytest

from atlas.scout.repo_scout import RepoScout


@pytest.mark.asyncio
async def test_repo_scout_detects_style_and_tests(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\naddopts = \"-q\"\n"
    )
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("def run():\n    return 1\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_app.py").write_text("def test_app():\n    assert True\n")

    scout = RepoScout()
    report = await scout.scan(tmp_path)

    assert report.languages.get(".py", 0) >= 2
    assert "pyproject.toml" in report.style_guides
    assert "tests/test_app.py" in report.test_files
