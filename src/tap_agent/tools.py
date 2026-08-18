from __future__ import annotations

import asyncio
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

from tap_agent.core_types import ToolCall, ToolDefinition, ToolResult

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
MAX_READ_LINES: int = 2000
MAX_WRITE_SIZE: int = 5 * 1024 * 1024  # 5 MB limit
BASH_TIMEOUT_SECONDS: int | float = 30


READ_TOOL_DEFINITION = ToolDefinition(
    name="read",
    description=(
        "Read the contents of a file inside the project directory. "
        "Optionally specify start_line and end_line (1-based, inclusive) "
        "to read only a slice of the file. "
        "Output is capped at MAX_READ_LINES lines."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file, relative to the project root.",
            },
            "start_line": {
                "type": "integer",
                "description": "First line to read (1-based, inclusive). Defaults to 1.",
            },
            "end_line": {
                "type": "integer",
                "description": "Last line to read (1-based, inclusive). Defaults to end of file.",
            },
        },
        "required": ["path"],
        "additionalProperties": False,
    },
)

BASH_TOOL_DEFINITION = ToolDefinition(
    name="bash",
    description=(
        "Run a read-only shell command inside the project directory. "
        "Dangerous commands (rm, del, format, shutdown, etc.) are blocked. "
        "Command output is returned as a string. "
        f"Timeout: {BASH_TIMEOUT_SECONDS}s."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Shell command to execute.",
            },
        },
        "required": ["command"],
        "additionalProperties": False,
    },
)

WRITE_TOOL_DEFINITION = ToolDefinition(
    name="write",
    description=(
        "Write content to a file inside the project directory. "
        "Supports 'overwrite' and 'append' modes. "
        "Automatically creates parent directories if they do not exist."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file, relative to the project root.",
            },
            "content": {
                "type": "string",
                "description": "Content to write to the file.",
            },
            "mode": {
                "type": "string",
                "enum": ["overwrite", "append"],
                "description": "Write mode. Defaults to 'overwrite'.",
            },
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    },
)

AVAILABLE_TOOLS: list[ToolDefinition] = [READ_TOOL_DEFINITION, WRITE_TOOL_DEFINITION, BASH_TOOL_DEFINITION]


class ToolExecutionError(RuntimeError):
    """Raised by _read_tool / _bash_tool to signal a handled error.

    execute_tool() always catches this and converts it into ToolResult.fail().
    """


def _resolve_within_project(raw_path: str) -> Path:
    """Resolve *raw_path* and ensure it stays inside PROJECT_ROOT.

    Raises ToolExecutionError on path-traversal attempts.
    """
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    resolved = candidate.resolve()

    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError:
        raise ToolExecutionError(
            f"Path traversal denied: '{raw_path}' resolves outside project root."
        )
    return resolved


# Dangerous command fragments – checked case-insensitively.
_UNIX_DANGEROUS: tuple[str, ...] = (
    "rm -rf",
    "rm -fr",
    "sudo ",
    "chmod -r",
    "mkfs",
    "dd if=",
    "> /dev/",
    "shred ",
    ":(){:|:&};:",  # fork bomb
)

_WINDOWS_DANGEROUS: tuple[str, ...] = (
    "del /",
    "rd /s",
    "rmdir /s",
    "format ",
    "shutdown",
    "remove-item",
    "ri ",          # PowerShell alias for Remove-Item
    "taskkill",
    "reg delete",
    "cipher /w",
    "diskpart",
    "bcdedit",
    "net user",
    "net localgroup",
)

_ALL_DANGEROUS: tuple[str, ...] = _UNIX_DANGEROUS + _WINDOWS_DANGEROUS


def _is_dangerous(command: str) -> bool:
    """Return True if *command* contains a known-dangerous substring."""
    lower = command.lower()
    return any(fragment in lower for fragment in _ALL_DANGEROUS)


def _get_bash_executable() -> str | None:
    """Tìm đường dẫn thực thi của Git Bash trên Windows (Bỏ qua WSL)."""
    if sys.platform != "win32":
        return None

    # 1. Tìm trong hệ thống PATH, nhưng LOẠI BỎ System32 (WSL)
    in_path = shutil.which("bash.exe") or shutil.which("bash")
    if in_path and "system32" not in in_path.lower():
        return in_path

    # 2. Tìm các đường dẫn cài đặt mặc định của Git Bash
    possible_paths = [
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\Program Files (x86)\Git\bin\bash.exe",
        os.path.expanduser(r"~\AppData\Local\Programs\Git\bin\bash.exe"),
    ]
    return next((p for p in possible_paths if os.path.exists(p)), None)

# ---------------------------------------------------------------------------
# Core tool implementations
# ---------------------------------------------------------------------------
def _read_tool(arguments: dict[str, Any]) -> str:
    """Synchronous file reader."""
    raw_path: str | None = arguments.get("path")
    if not raw_path:
        raise ToolExecutionError("Missing required argument: 'path'.")

    resolved = _resolve_within_project(raw_path)

    if not resolved.exists():
        raise ToolExecutionError(f"File not found: '{raw_path}'.")
    if not resolved.is_file():
        raise ToolExecutionError(f"Path is not a file: '{raw_path}'.")

    try:
        text = resolved.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise ToolExecutionError(
            f"Cannot read '{raw_path}': binary file detected. Only UTF-8 text files are supported."
        )

    lines = text.splitlines(keepends=True)
    total_lines = len(lines)

    start_line: int = int(arguments.get("start_line") or 1)
    end_line: int = int(arguments.get("end_line") or total_lines)

    start_line = max(1, start_line)
    end_line = min(total_lines, end_line)

    if start_line > end_line:
        return ""

    sliced = lines[start_line - 1 : end_line]

    truncated = False
    if len(sliced) > MAX_READ_LINES:
        sliced = sliced[:MAX_READ_LINES]
        truncated = True

    content = "".join(sliced)
    if truncated:
        content += f"\n... truncated at {MAX_READ_LINES} lines"

    return content


def _write_tool(arguments: dict[str, Any]) -> str:
    """Synchronous file writer."""
    raw_path: str | None = arguments.get("path")
    content: str | None = arguments.get("content")
    mode: str = arguments.get("mode", "overwrite")

    if not raw_path:
        raise ToolExecutionError("Missing required argument: 'path'.")
    if content is None:
        raise ToolExecutionError("Missing required argument: 'content'.")
    
    if mode not in ("overwrite", "append"):
        raise ToolExecutionError(f"Invalid mode: '{mode}'. Must be 'overwrite' or 'append'.")

    # Limit maximum write size
    content_bytes = content.encode("utf-8")
    actual_bytes = len(content_bytes)
    if actual_bytes > MAX_WRITE_SIZE:
        raise ToolExecutionError(f"Content exceeds maximum write size of {MAX_WRITE_SIZE} bytes.")

    resolved = _resolve_within_project(raw_path)

    # Automatically create parent directories
    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        raise ToolExecutionError(f"Permission denied: Cannot create parent directories for '{raw_path}'.")
    except OSError as e:
        raise ToolExecutionError(f"OS error when creating parent directories: {e}")

    file_mode = "a" if mode == "append" else "w"
    existed = resolved.exists()

    try:
        with resolved.open(file_mode, encoding="utf-8") as f:
            f.write(content)
    except PermissionError:
        raise ToolExecutionError(f"Permission denied: Cannot write to '{raw_path}'.")
    except OSError as e:
        raise ToolExecutionError(f"OS error when writing to file: {e}")

    if mode == "append":
        action = "Appended to"
    elif existed:
        action = "Overwrote"
    else:
        action = "Created new"

    return f"{action} file '{raw_path}' successfully ({actual_bytes} bytes)."


async def _bash_tool(arguments: dict[str, Any]) -> str:
    """Async shell executor đa nền tảng.

    - Đã tương thích với Windows (ưu tiên Git Bash, fallback sang PowerShell).
    - Tự động diệt sạch process tree khi xảy ra Timeout.
    """
    command: str | None = arguments.get("command")
    if not command or not command.strip():
        raise ToolExecutionError("Missing required argument: 'command'.")

    if _is_dangerous(command):
        raise ToolExecutionError(
            f"Command blocked for safety: '{command}'. "
            "Dangerous operations (rm, del, format, shutdown, etc.) are not allowed."
        )

    # Cấu hình Executable dựa trên Hệ điều hành
    if sys.platform == "win32":
        bash_path = _get_bash_executable()
        if bash_path:
            # Ưu tiên chạy qua Git Bash nếu máy có cài đặt
            proc = await asyncio.create_subprocess_exec(
                bash_path,
                "-c",
                command,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        else:
            # Fallback sang PowerShell nếu không tìm thấy Git Bash
            proc = await asyncio.create_subprocess_exec(
                "powershell.exe",
                "-Command",
                command,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
    else:
        # Linux / macOS: Chạy shell Unix tiêu chuẩn
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )

    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(),
            timeout=BASH_TIMEOUT_SECONDS,
        )
    except (asyncio.TimeoutError, TimeoutError):
        # Dọn dẹp tiến trình triệt để khi Timeout
        if sys.platform == "win32":
            os.system(f"taskkill /F /T /PID {proc.pid} >nul 2>&1")
        else:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass

        raise ToolExecutionError(
            f"Command timed out after {BASH_TIMEOUT_SECONDS}s and was killed: '{command}'."
        )

    stdout = stdout_bytes.decode("utf-8", errors="replace")
    stderr = stderr_bytes.decode("utf-8", errors="replace")

    exit_code = proc.returncode

    if exit_code != 0:
        error_msg = stderr.strip() or stdout.strip() or f"exit code {exit_code}"
        raise ToolExecutionError(
            f"Command failed (exit code {exit_code}): {error_msg}"
        )

    return stdout


# --- KỊCH BẢN CHẠY TEST ---
async def main():
    print("=== BẮT ĐẦU TEST CHẠY THỬ ===\n")

    # Case 1: Lệnh hợp lệ (Chạy tốt lệnh ls -la trên mọi OS)
    print("--- Test 1: Lệnh liệt kê thư mục (ls -la) ---")
    try:
        result = await _bash_tool({"command": "ls -la"})
        print(f"[Thành công] Output:\n{result}")
    except ToolExecutionError as e:
        print(f"[Lỗi] {e}")

    # Case 2: Lệnh nguy hiểm (bị block)
    print("--- Test 2: Lệnh nguy hiểm (rm -rf .) ---")
    try:
        result = await _bash_tool({"command": "rm -rf ."})
        print(f"[Thành công] Output:\n{result}")
    except ToolExecutionError as e:
        print(f"[Bắt lỗi thành công] {e}\n")

    # Case 3: Lệnh sai / không tồn tại
    print("--- Test 3: Lệnh không tồn tại ---")
    try:
        result = await _bash_tool({"command": "non_existent_command_123"})
        print(f"[Thành công] Output:\n{result}")
    except ToolExecutionError as e:
        print(f"[Bắt lỗi thành công] {e}\n")


# ---------------------------------------------------------------------------
# Dispatch table & public execute_tool()
# ---------------------------------------------------------------------------
def _run_bash(arguments: dict[str, Any]) -> Awaitable[str]:
    """Bridge to async _bash_tool."""
    return _bash_tool(arguments)


_HANDLERS: dict[str, Any] = {
    "read": _read_tool,
    "write": _write_tool,
    "bash": _run_bash,
}


async def execute_tool(tool_call: ToolCall) -> ToolResult:
    """Dispatch *tool_call* to the matching handler and always return a ToolResult."""
    call_id = tool_call.id

    handler = _HANDLERS.get(tool_call.name)
    if handler is None:
        return ToolResult.fail(call_id, f"unknown tool: '{tool_call.name}'.")

    try:
        output_or_coro = handler(tool_call.arguments)
        if asyncio.iscoroutine(output_or_coro):
            output = await output_or_coro
        else:
            output = output_or_coro
        return ToolResult.ok(call_id, str(output))

    except ToolExecutionError as exc:
        return ToolResult.fail(call_id, str(exc))

    except Exception as exc:  # noqa: BLE001
        return ToolResult.fail(
            call_id,
            f"Unexpected error in tool '{tool_call.name}': {type(exc).__name__}: {exc}",
        )


if __name__ == "__main__":
    asyncio.run(main())