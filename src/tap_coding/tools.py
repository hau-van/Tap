import asyncio
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

BASH_MAX_OUTPUT_BYTES = 50 * 1024


def get_bash_executable() -> str | None:
    if os.name != "nt":
        return None

    possible_paths = [
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\Program Files (x86)\Git\bin\bash.exe",
        os.path.expanduser(r"~\AppData\Local\Programs\Git\bin\bash.exe"),
    ]
    return next((p for p in possible_paths if os.path.exists(p)), None)


def create_bash_tool(cwd: str | Path = "."): # -> AgentTool:
    _cwd = Path(cwd).resolve()
    git_bash_path = get_bash_executable()
    # print(_cwd)
    # print(git_bash_path)
    # return

    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Shell command to run.",
            },
            "timeout": {
                "type": "number",
                "description": "Timeout in seconds (default: 30s).",
            },
        },
        "required": ["command"],
    }

    async def executor(tc):
        command = str(tc.arguments.get("command", "")).strip()
        if not command:
            return {
                "tool_call_id": tc.id,
                "name": "bash",
                "ok": False,
                "content": "Error: Empty command.",
            }

        timeout = tc.arguments.get("timeout")
        try:
            timeout_val = float(timeout) if timeout is not None else 30.0
        except (ValueError, TypeError):
            timeout_val = 30.0

        start_time = time.monotonic()
        timed_out = False

        try:
            if os.name == "nt" and git_bash_path:
                proc = await asyncio.create_subprocess_exec(
                    git_bash_path,
                    "-c",
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    cwd=str(_cwd),
                )
            else:
                proc = await asyncio.create_subprocess_shell(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    cwd=str(_cwd),
                    start_new_session=(os.name != "nt"),
                )

            try:
                stdout_bytes, _ = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout_val
                )
                exit_code = (
                    proc.returncode if proc.returncode is not None else 0
                )
            except TimeoutError:
                timed_out = True
                exit_code = -1
                if os.name != "nt":
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    # Diệt tiến trình trên Windows kèm cây tiến trình con
                    os.system(f"taskkill /F /T /PID {proc.pid} >nul 2>&1")

                stdout_bytes, _ = await proc.communicate()

        except Exception as exc:  # noqa: BLE001
            return {
                "tool_call_id": tc.id,
                "name": "bash",
                "ok": False,
                "content": f"Process error: {exc!s}",
            }

        duration = time.monotonic() - start_time
        output = stdout_bytes.decode("utf-8", errors="replace")

        # Giới hạn kích thước output
        if len(output.encode("utf-8")) > BASH_MAX_OUTPUT_BYTES:
            output = (
                output.encode("utf-8")[:BASH_MAX_OUTPUT_BYTES].decode(
                    "utf-8", errors="ignore"
                )
                + "\n... (output truncated)"
            )

        header = f"exit_code={exit_code} | duration={duration:.2f}s"
        if timed_out:
            header += " | TIMED OUT"

        return {
            "tool_call_id": tc.id,
            "name": "bash",
            "ok": (exit_code == 0 and not timed_out),
            "content": f"<bash_output>\n# {header}\n{output}\n</bash_output>",
        }

    return {
        "name": "bash",
        "description": "Execute shell commands and return combined stdout/stderr.",
        "input_schema": input_schema,
        "executor": executor,
    }

if __name__ == "__main__":
    print(os.name)
    print(get_bash_executable())
    create_bash_tool()