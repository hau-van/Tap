"""Core provider-neutral types cho tap_agent.

Không import gì ngoài stdlib + pydantic. Không biết OpenAI/Gemini/filesystem là gì.
Mọi layer khác của agent (provider, loop, tools, cli) đều import từ đây.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Union

from pydantic import BaseModel, ConfigDict, Field


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


# ─── Tool primitives ─────────────────────────────────────────

class ToolCall(BaseModel):
    """Assistant's request to invoke a tool.

    Được LLM sinh ra trong assistant turn. Loop sẽ tra `name` trong tool
    registry, execute với `arguments`, rồi tạo ToolResult tương ứng.
    """
    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    arguments: dict


class ToolResult(BaseModel):
    """Kết quả sau khi execute một ToolCall.

    Ghép cặp với ToolCall qua `tool_call_id`. `success=False` KHÔNG raise
    exception ra ngoài loop — nó là dữ liệu bình thường để LLM đọc và tự sửa.
    """
    model_config = ConfigDict(frozen=True)

    tool_call_id: str
    success: bool
    output: str
    error: str | None = None


class ToolDefinition(BaseModel):
    """Schema của một tool được expose cho LLM.

    Chỉ chứa metadata mà LLM cần biết để quyết định gọi tool. Không chứa
    executor — implementation nằm ở layer khác.
    """
    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    parameters_schema: dict   # JSON Schema draft-07, dùng cho function calling


# ─── Message ─────────────────────────────────────────────────

class Message(BaseModel):
    """Một message trong transcript conversation.

    Dùng chung cho user / assistant / tool. `tool_calls` chỉ có ở
    role="assistant"; role="tool" thì `content` chứa tool output đã stringify.
    """
    model_config = ConfigDict(frozen=True)

    role: Literal["user", "assistant", "tool"]
    content: str
    tool_calls: list[ToolCall] | None = None
    timestamp: datetime = Field(default_factory=_utc_now)


# ─── Agent event ─────────────────────────────────────────────

class AgentEvent(BaseModel):
    """Sự kiện do agent runtime emit ra event stream.

    - type="message":     payload là Message vừa được append vào transcript
    - type="tool_call":   payload là ToolCall LLM vừa yêu cầu
    - type="tool_result": payload là ToolResult vừa execute xong
    - type="error":       payload là str mô tả lỗi
    - type="done":        payload là str (có thể rỗng), báo agent kết thúc run
    """
    model_config = ConfigDict(frozen=True)

    type: Literal["message", "tool_call", "tool_result", "error", "done"]
    payload: Union[Message, ToolCall, ToolResult, str]
    