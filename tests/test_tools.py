"""
tests/test_tools.py – Phase 5

Unit tests for tap_agent.tools (read tool + bash tool + execute_tool dispatcher).

Rules:
- NO filesystem mocking – all tests use real files and real processes.
- tmp_project fixture creates files INSIDE PROJECT_ROOT to pass the
  path-traversal guard while remaining isolated.
- Timeout test uses a real subprocess (sleep / ping) with BASH_TIMEOUT_SECONDS
  monkeypatched to 0.2 s.
"""
from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

import pytest

import tap_agent.tools as tools_mod
from tap_agent.core_types import ToolCall, ToolResult
from tap_agent.tools import (
    AVAILABLE_TOOLS,
    BASH_TOOL_DEFINITION,
    READ_TOOL_DEFINITION,
    execute_tool,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_call(name: str, **kwargs) -> ToolCall:
    """Convenience: build a ToolCall with a synthetic ID."""
    return ToolCall(id=f"call_{uuid.uuid4().hex[:8]}", name=name, arguments=kwargs)


# ---------------------------------------------------------------------------
# Fixture: tmp_project
# ---------------------------------------------------------------------------
@pytest.fixture()
def tmp_project(tmp_path_factory):
    """
    Create a temporary directory INSIDE PROJECT_ROOT so that
    _resolve_within_project() does not reject legitimate test paths.

    Yields the Path to the temp directory and cleans up afterwards.
    """
    # Place the temp dir under PROJECT_ROOT / "_test_tmp_<uid>"
    uid = uuid.uuid4().hex[:8]
    temp_dir: Path = tools_mod.PROJECT_ROOT / f"_test_tmp_{uid}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================
# READ TOOL TESTS
# ============================================================


class TestReadTool:
    # ----------------------------------------------------------
    # Test 1: Read entire file successfully
    # ----------------------------------------------------------
    @pytest.mark.asyncio
    async def test_read_entire_file(self, tmp_project: Path):
        """ToolResult.ok with full file content."""
        target = tmp_project / "hello.txt"
        target.write_text("line1\nline2\nline3\n", encoding="utf-8")

        # Pass path relative to PROJECT_ROOT
        rel = target.relative_to(tools_mod.PROJECT_ROOT)
        result = await execute_tool(_make_call("read", path=str(rel)))

        assert isinstance(result, ToolResult)
        assert result.success is True
        assert "line1" in result.output
        assert "line2" in result.output
        assert "line3" in result.output

    # ----------------------------------------------------------
    # Test 2: Read with start_line / end_line
    # ----------------------------------------------------------
    @pytest.mark.asyncio
    async def test_read_line_slice(self, tmp_project: Path):
        """Only lines in [start_line, end_line] are returned."""
        target = tmp_project / "numbered.txt"
        target.write_text("\n".join(f"line{i}" for i in range(1, 11)), encoding="utf-8")

        rel = target.relative_to(tools_mod.PROJECT_ROOT)
        result = await execute_tool(_make_call("read", path=str(rel), start_line=3, end_line=5))

        assert result.success is True
        assert "line3" in result.output
        assert "line5" in result.output
        # Lines outside the slice must not appear
        assert "line1" not in result.output
        assert "line6" not in result.output

    # ----------------------------------------------------------
    # Test 3: Truncation when file exceeds MAX_READ_LINES
    # ----------------------------------------------------------
    @pytest.mark.asyncio
    async def test_read_truncates_long_file(self, tmp_project: Path, monkeypatch):
        """Output is capped at MAX_READ_LINES; truncation note is appended."""
        monkeypatch.setattr(tools_mod, "MAX_READ_LINES", 5)

        target = tmp_project / "big.txt"
        target.write_text("\n".join(f"row{i}" for i in range(1, 21)), encoding="utf-8")

        rel = target.relative_to(tools_mod.PROJECT_ROOT)
        result = await execute_tool(_make_call("read", path=str(rel)))

        assert result.success is True
        assert "row5" in result.output
        assert "row6" not in result.output
        assert "truncated at 5 lines" in result.output

    # ----------------------------------------------------------
    # Test 4: Non-existent file → ToolResult.fail
    # ----------------------------------------------------------
    @pytest.mark.asyncio
    async def test_read_nonexistent_file(self, tmp_project: Path):
        """Missing file produces ToolResult.fail (no exception raised)."""
        rel = tmp_project.relative_to(tools_mod.PROJECT_ROOT) / "ghost.txt"
        result = await execute_tool(_make_call("read", path=str(rel)))

        assert isinstance(result, ToolResult)
        assert result.success is False
        assert result.error is not None
        assert "not found" in result.error.lower() or "ghost" in result.error

    # ----------------------------------------------------------
    # Test 5: Path traversal → ToolResult.fail
    # ----------------------------------------------------------
    @pytest.mark.asyncio
    async def test_read_path_traversal_blocked(self):
        """Path outside PROJECT_ROOT is denied without raising."""
        result = await execute_tool(_make_call("read", path="../../etc/passwd"))

        assert isinstance(result, ToolResult)
        assert result.success is False
        assert result.error is not None
        assert "traversal" in result.error.lower() or "denied" in result.error.lower()

    # ----------------------------------------------------------
    # Test 6: Binary file → ToolResult.fail
    # ----------------------------------------------------------
    @pytest.mark.asyncio
    async def test_read_binary_file_refused(self, tmp_project: Path):
        """Binary content triggers UnicodeDecodeError → ToolResult.fail."""
        target = tmp_project / "data.bin"
        target.write_bytes(bytes(range(256)))  # 256 bytes, definitely not UTF-8

        rel = target.relative_to(tools_mod.PROJECT_ROOT)
        result = await execute_tool(_make_call("read", path=str(rel)))

        assert isinstance(result, ToolResult)
        assert result.success is False
        assert result.error is not None
        assert "binary" in result.error.lower()

    # ----------------------------------------------------------
    # Test 7: Missing 'path' argument → ToolResult.fail
    # ----------------------------------------------------------
    @pytest.mark.asyncio
    async def test_read_missing_path_argument(self):
        """ToolCall without 'path' key → ToolResult.fail (not an exception)."""
        result = await execute_tool(_make_call("read"))  # no path kwarg

        assert isinstance(result, ToolResult)
        assert result.success is False
        assert result.error is not None
        assert "path" in result.error.lower()


# ============================================================
# BASH TOOL TESTS
# ============================================================


class TestBashTool:
    # ----------------------------------------------------------
    # Test 8: Basic echo command succeeds
    # ----------------------------------------------------------
    @pytest.mark.asyncio
    async def test_bash_echo_succeeds(self):
        """Simple echo → ToolResult.ok with output."""
        if os.name == "nt":
            cmd = "echo hello_world"
        else:
            cmd = "echo hello_world"

        result = await execute_tool(_make_call("bash", command=cmd))

        assert isinstance(result, ToolResult)
        assert result.success is True
        assert "hello_world" in result.output

    # ----------------------------------------------------------
    # Test 9: Dangerous command is blocked
    # ----------------------------------------------------------
    @pytest.mark.asyncio
    async def test_bash_dangerous_command_blocked(self):
        """rm -rf / is blocked before any subprocess is created."""
        result = await execute_tool(_make_call("bash", command="rm -rf /"))

        assert isinstance(result, ToolResult)
        assert result.success is False
        assert result.error is not None
        assert "blocked" in result.error.lower() or "dangerous" in result.error.lower()

    @pytest.mark.asyncio
    async def test_bash_windows_dangerous_command_blocked(self):
        """del / is blocked (Windows blacklist applies on all platforms)."""
        result = await execute_tool(_make_call("bash", command="del / some_path"))

        assert isinstance(result, ToolResult)
        assert result.success is False
        assert result.error is not None

    # ----------------------------------------------------------
    # Test 10: Failed command (exit code != 0) → ToolResult.fail
    # ----------------------------------------------------------
    @pytest.mark.asyncio
    async def test_bash_nonzero_exit_code(self):
        """A command that exits non-zero → ToolResult.fail with stderr/stdout."""
        if os.name == "nt":
            # 'dir' on a path that doesn't exist exits non-zero on Windows
            cmd = "dir C:\\this_path_does_not_exist_xyz_abc_123"
        else:
            cmd = "ls /this_path_does_not_exist_xyz_abc_123"

        result = await execute_tool(_make_call("bash", command=cmd))

        assert isinstance(result, ToolResult)
        assert result.success is False
        assert result.error is not None

    # ----------------------------------------------------------
    # Test 11: Real timeout – process is killed within BASH_TIMEOUT_SECONDS
    # ----------------------------------------------------------
    @pytest.mark.asyncio
    async def test_bash_timeout_kills_process(self, monkeypatch):
        """
        Monkeypatch BASH_TIMEOUT_SECONDS to 0.2 s and run a long-running
        command.  The result must be ToolResult.fail mentioning timeout,
        and the test itself must complete well under 5 s (proving the kill
        works).
        """
        monkeypatch.setattr(tools_mod, "BASH_TIMEOUT_SECONDS", 0.2)

        if os.name == "nt":
            # 'ping -n 6 127.0.0.1' waits ~5 seconds on Windows
            cmd = "ping -n 6 127.0.0.1"
        else:
            cmd = "sleep 5"

        result = await execute_tool(_make_call("bash", command=cmd))

        assert isinstance(result, ToolResult)
        assert result.success is False
        assert result.error is not None
        assert "timed out" in result.error.lower() or "timeout" in result.error.lower()

    # ----------------------------------------------------------
    # Test 12: Missing 'command' argument → ToolResult.fail
    # ----------------------------------------------------------
    @pytest.mark.asyncio
    async def test_bash_missing_command_argument(self):
        """ToolCall without 'command' key → ToolResult.fail (not an exception)."""
        result = await execute_tool(_make_call("bash"))  # no command kwarg

        assert isinstance(result, ToolResult)
        assert result.success is False
        assert result.error is not None
        assert "command" in result.error.lower()


# ============================================================
# EXECUTE_TOOL DISPATCHER TESTS
# ============================================================


class TestExecuteTool:
    # ----------------------------------------------------------
    # Test 13: Unknown tool name → ToolResult.fail, no exception
    # ----------------------------------------------------------
    @pytest.mark.asyncio
    async def test_unknown_tool_returns_fail(self):
        """execute_tool with an unknown name returns ToolResult.fail."""
        result = await execute_tool(_make_call("nonexistent_tool_xyz"))

        assert isinstance(result, ToolResult)
        assert result.success is False
        assert result.error is not None
        assert "unknown tool" in result.error.lower()

    # ----------------------------------------------------------
    # Test 14: execute_tool never raises – even with a broken handler
    # ----------------------------------------------------------
    @pytest.mark.asyncio
    async def test_execute_tool_never_raises(self, monkeypatch):
        """Even if a handler throws an unexpected exception, execute_tool
        swallows it and returns ToolResult.fail."""

        def _exploding_handler(_args):
            raise RuntimeError("Simulated unexpected crash!")

        monkeypatch.setitem(tools_mod._HANDLERS, "read", _exploding_handler)

        result = await execute_tool(_make_call("read", path="anything.py"))

        assert isinstance(result, ToolResult)
        assert result.success is False
        assert result.error is not None
        # Must NOT propagate the RuntimeError


# ============================================================
# SANITY: AVAILABLE_TOOLS export
# ============================================================


class TestAvailableTools:
    def test_available_tools_contains_both(self):
        """AVAILABLE_TOOLS exports exactly read and bash."""
        names = {t.name for t in AVAILABLE_TOOLS}
        assert names == {"read", "bash"}

    def test_tool_definitions_have_required_fields(self):
        """Each ToolDefinition has name, description, parameters_schema."""
        for td in AVAILABLE_TOOLS:
            assert td.name
            assert td.description
            assert isinstance(td.parameters_schema, dict)
            assert "properties" in td.parameters_schema
            assert "required" in td.parameters_schema
