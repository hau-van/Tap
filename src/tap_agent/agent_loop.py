
from __future__ import annotations
 
import asyncio

import json

import pathlib

from collections.abc import Callable
 

from tap_agent.core_types import (

	AgentEvent,

	Message,

	ToolCall,

	ToolDefinition,

	ToolResult,

)

from tap_agent.provider import MockProvider, ModelProvider
 
async def run_agent_loop(
    messages: list[Message],
    tools: list[ToolDefinition],
    provider: ModelProvider,
    max_iterations: int = 10,
):
    messages = list(messages)
    for _ in range(max_iterations):
        reply = await provider.generate_reply(messages, tools)
        messages.append(reply)
        if reply.tool_calls:
            for tool_call in reply.tool_calls:
                tool_result = yield AgentEvent.tool_call(tool_call)
                if tool_result is not None:
                    if not isinstance(tool_result, ToolResult):
                        yield AgentEvent.error(f"expected ToolResult, got {type(tool_result)}")
                        return
                    tool_message = Message(
                        role="tool",
                        content=json.dumps({
                            "name": tool_call.name,
                            "id": tool_call.id,
                            "success": tool_result.success,
                            "output": tool_result.output,
                            "error": tool_result.error,
                        })
                    )
                    messages.append(tool_message)
        else:
            yield AgentEvent.message(reply)
            yield AgentEvent.done()
            return
            
    yield AgentEvent.error("max iterations reached")

 
READ_TOOL = ToolDefinition(

	name="read",

	description="Read a file",

	parameters_schema={"type": "object", "properties": {"path": {"type": "string"}}},

)
 
 
class RecordingProvider(MockProvider):

	"""Wraps MockProvider to record every call, so tests can assert on

	exactly what messages the loop sent back to the provider (e.g. to

	verify the tool-result message was appended correctly)."""
 
	def __init__(self, script: list[Message]) -> None:

		super().__init__(script)

		self.calls: list[list[Message]] = []
 
	async def generate_reply(self, messages, tools):  # type: ignore[override]

		self.calls.append(list(messages))

		return await super().generate_reply(messages, tools)
 
 
async def _drive(

	messages: list[Message],

	tools: list[ToolDefinition],

	provider,

	*,

	max_iterations: int = 10,

	tool_result_factory: Callable[[ToolCall], ToolResult] | None = None,

) -> list[AgentEvent]:

	"""Drive run_agent_loop() to completion, auto-answering every

	tool_call with `tool_result_factory` (default: always succeeds)."""

	tool_result_factory = tool_result_factory or (

		lambda call: ToolResult.ok(call.id, "file contents")

	)

	gen = run_agent_loop(messages, tools, provider, max_iterations=max_iterations)

	events: list[AgentEvent] = []

	sent: ToolResult | None = None

	while True:

		try:

			event = await gen.asend(sent)

		except StopAsyncIteration:

			break

		events.append(event)

		sent = tool_result_factory(event.payload) if event.type == "tool_call" else None

	return events
 
 
# ---------------------------------------------------------------------

# 1. Prompt -> ToolCall / 2. Resume via asend / 3. Final Message / 4. done()

# ---------------------------------------------------------------------

def test_tool_call_then_resume_then_final_message_and_done():

	tool_call = ToolCall(id="call_1", name="read", arguments={"path": "a.py"})

	script = [

		Message(role="assistant", content="", tool_calls=[tool_call]),

		Message(role="assistant", content="Here is the summary."),

	]

	provider = RecordingProvider(script)

	first_user_message = Message(role="user", content="Summarize a.py")

	original_messages = [first_user_message]
 
	events = asyncio.run(_drive(original_messages, [READ_TOOL], provider))
 
	# Event sequence: tool_call -> message -> done

	assert [e.type for e in events] == ["tool_call", "message", "done"]
 
	# 1. Prompt -> ToolCall

	assert events[0].payload == tool_call
 
	# 3. Final assistant Message

	assert events[1].payload.content == "Here is the summary."

	assert events[1].payload.role == "assistant"
 
	# 4. AgentEvent.done()

	assert events[2].type == "done"

	assert events[2].payload is None
 
	# The original `messages` list passed in must never be mutated.

	assert original_messages == [first_user_message]

	assert len(original_messages) == 1
 
	# 2. Resume via asend(ToolResult.ok(...)): the *second* provider call

	# must include the tool-role message built from what was asend()'d.

	assert len(provider.calls) == 2

	second_call_messages = provider.calls[1]

	tool_message = second_call_messages[-1]

	assert tool_message.role == "tool"

	payload = json.loads(tool_message.content)

	assert payload == {

		"name": "read",

		"id": "call_1",

		"success": True,

		"output": "file contents",

		"error": None,

	}
 
 
def test_multiple_tool_calls_in_one_turn_are_each_resumed_separately():

	call_a = ToolCall(id="call_a", name="read", arguments={"path": "a.py"})

	call_b = ToolCall(id="call_b", name="read", arguments={"path": "b.py"})

	script = [

		Message(role="assistant", content="", tool_calls=[call_a, call_b]),

		Message(role="assistant", content="Done reading both files."),

	]

	provider = RecordingProvider(script)
 
	events = asyncio.run(

		_drive([Message(role="user", content="Read a.py and b.py")], [READ_TOOL], provider)

	)
 
	assert [e.type for e in events] == ["tool_call", "tool_call", "message", "done"]

	assert events[0].payload == call_a

	assert events[1].payload == call_b
 
	# Both tool-result messages must have been appended before the model

	# was asked again.

	second_call_messages = provider.calls[1]

	tool_messages = [m for m in second_call_messages if m.role == "tool"]

	assert len(tool_messages) == 2

	assert json.loads(tool_messages[0].content)["id"] == "call_a"

	assert json.loads(tool_messages[1].content)["id"] == "call_b"
 
 
# ---------------------------------------------------------------------

# 5. max_iterations reached

# ---------------------------------------------------------------------

def test_max_iterations_reached():

	# The loop checks the limit *before* each provider call, so exactly

	# `max_iterations` calls happen — the script must cover all of them.

	tool_call = ToolCall(id="call_x", name="read", arguments={"path": "x.py"})

	script = [Message(role="assistant", content="", tool_calls=[tool_call]) for _ in range(3)]

	provider = RecordingProvider(script)
 
	events = asyncio.run(

		_drive(

			[Message(role="user", content="loop forever")],

			[READ_TOOL],

			provider,

			max_iterations=3,

		)

	)
 
	assert events[-1].type == "error"

	assert events[-1].payload == "max iterations reached"

	# Exactly max_iterations provider calls, never a 4th.

	assert len(provider.calls) == 3
 
 
# ---------------------------------------------------------------------

# Extra: resume contract — sending the wrong type must not silently pass.

# ---------------------------------------------------------------------

def test_wrong_asend_type_yields_error_instead_of_crashing():

	tool_call = ToolCall(id="call_1", name="read", arguments={"path": "a.py"})

	script = [Message(role="assistant", content="", tool_calls=[tool_call])]

	provider = RecordingProvider(script)
 
	async def _run() -> list[AgentEvent]:

		gen = run_agent_loop([Message(role="user", content="hi")], [READ_TOOL], provider)

		events = [await gen.asend(None)]

		# Send a plain string instead of a ToolResult.

		events.append(await gen.asend("not a ToolResult"))  # type: ignore[arg-type]

		return events
 
	events = asyncio.run(_run())

	assert events[0].type == "tool_call"

	assert events[1].type == "error"
 
 
# ---------------------------------------------------------------------

# Acceptance criteria: no forbidden imports.

# ---------------------------------------------------------------------

def test_does_not_import_tools_or_concrete_providers():

	"""Only real `import`/`from` lines are checked — a docstring may still

	*mention* e.g. GeminiProvider in prose as an illustrative example."""

	source = pathlib.Path(__file__).read_text()

	import_lines = [

		line.strip()

		for line in source.splitlines()

		if line.strip().startswith(("import ", "from "))

	]

	forbidden = ["tap_agent.tools", "gemini_provider", "google.genai", "google_genai"]

	for line in import_lines:

		for token in forbidden:

			assert token not in line, f"forbidden import found: {line!r}"
 