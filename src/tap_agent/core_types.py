
from __future__ import annotations
import uuid
from datetime import datetime, timezone
from typing import Any, Literal, Optional, Union
from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Helpers & Type Aliases
# ---------------------------------------------------------------------------
Role = Literal["user", "assistant", "tool"]
EventType = Literal["message", "tool_call", "tool_result", "error", "done"]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Tool Primitives
# ---------------------------------------------------------------------------
class ToolDefinition(BaseModel):
    """Schema của một tool được expose cho LLM.
    
    Chỉ chứa metadata mà LLM cần biết. Việc wrap schema này thành format
    của OpenAI hay Gemini sẽ do layer Provider đảm nhiệm.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    description: str
    parameters_schema: dict[str, Any]


class ToolCall(BaseModel):
    """Yêu cầu gọi tool từ LLM (Assistant turn)."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    # Mặc định tự sinh ID, nhưng cho phép provider ghi đè (vd: OpenAI trả về call_id)
    id: str = Field(default_factory=lambda: _new_id("call"))
    name: str
    arguments: dict[str, Any]


class ToolResult(BaseModel):
    """Kết quả thực thi của một ToolCall."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    tool_call_id: str
    success: bool
    output: str
    error: Optional[str] = None

    @classmethod
    def ok(cls, tool_call_id: str, output: str) -> "ToolResult":
        return cls(tool_call_id=tool_call_id, success=True, output=output)

    @classmethod
    def fail(cls, tool_call_id: str, error: str) -> "ToolResult":
        return cls(tool_call_id=tool_call_id, success=False, output="", error=error)


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------
class Message(BaseModel):
    """Một message chuẩn trong transcript conversation.
    
    Thiết kế dạng "Flat": nếu role="tool", kết quả trả về chỉ là string 
    được lưu thẳng vào `content`. Cấu trúc này dễ serialize và tương thích tự 
    nhiên với mọi LLM Provider.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: _new_id("msg"))
    role: Role
    content: str
    tool_calls: Optional[list[ToolCall]] = None
    timestamp: datetime = Field(default_factory=_utc_now)


# ---------------------------------------------------------------------------
# Agent Event Stream
# ---------------------------------------------------------------------------
class AgentEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    type: EventType
    payload: Union[Message, ToolCall, ToolResult, str, None] = None

    @classmethod
    def message(cls, msg: Message) -> "AgentEvent":
        return cls(type="message", payload=msg)

    @classmethod
    def tool_call(cls, call: ToolCall) -> "AgentEvent":
        return cls(type="tool_call", payload=call)

    @classmethod
    def tool_result(cls, result: ToolResult) -> "AgentEvent":
        return cls(type="tool_result", payload=result)

    @classmethod
    def error(cls, message: str) -> "AgentEvent":
        return cls(type="error", payload=message)

    @classmethod
    def done(cls) -> "AgentEvent":
        return cls(type="done", payload=None)
