"""The agent loop built directly on the OpenAI SDK (no LangChain / LangGraph).

This module owns:
- the system prompt
- the ReAct-style loop with tool calls
- iteration limiting (to prevent runaway loops)
- event emission so the UI can render progress
- the :class:`Agent` object passed around by the CLI
- persistent memory integration with automatic summarization
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from my_agent.config.settings import settings
from my_agent.events import AgentEvent, AssistantEvent, ToolCallEvent, ToolResultEvent
from my_agent.memory import DEFAULT_MAX_CONTEXT_TOKENS, count_message_tokens, count_tokens, get_store
from my_agent.messages import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolCall,
    ToolMessage,
    Usage,
    UserMessage,
    from_openai,
    to_openai,
)
from my_agent.tools import CODING_TOOLS
from my_agent.tools._common import workspace_root

# Stop the tool loop after this many model turns, even if the model keeps asking
# for more tools. This prevents runaway loops on hard or impossible tasks.
MAX_ITERATIONS = 25

# Each model call is one "turn" in our loop.
DEFAULT_RECURSION_LIMIT = MAX_ITERATIONS


def build_system_prompt() -> str:
    """System prompt, rebuilt each turn so it reflects the current workspace."""
    workspace = workspace_root()
    return f"""\
You are PyPilot, a friendly and helpful terminal coding agent. You work inside a single workspace directory
and help the user inspect, edit, run and debug code.

Workspace: {workspace}
Operating system: {os.name}
Commands run in the workspace root with no TTY, so use non-interactive flags.

CRITICAL INSTRUCTIONS:
- If the user says "hello", "hi", "hey", "howdy", or similar greetings, respond naturally and conversationally like a human would. For example: "Hey! How can I help you today?" Do NOT say "done", "ok", or "completed".
- If the user asks a general question or just wants to chat with NO coding task, answer directly without calling ANY tools.
- NEVER respond with just "done", "ok", "okay", or similar one-word completion phrases for conversational messages.
- Only use tools when the user explicitly asks you to do something with code/files/system.

How you operate:
1. For greetings and casual conversation: Answer directly and naturally. Do not explore files or use tools.
2. For coding tasks: Explore before acting: use list_files/glob_files to find files and search_files to locate code. Read only what you need.
3. Read before you edit: always read_file a file before changing it.
4. Prefer edit_file for small, targeted changes (make old_str unique). Use write_file only for new files or full rewrites.
5. Verify your work: after changing code, run the relevant tests or commands with run_command and fix any failures. Never claim success without checking.
6. Work in small steps and keep calling tools until the task is done.

Rules:
- All paths are relative to the workspace root.
- Never invent tool names or arguments.
- For coding tasks you complete, reply with a short summary of what you changed and how you verified it.
- For casual messages or questions, just give a natural, helpful answer."""


@dataclass
class AgentResult:
    """Outcome of a single :meth:`Agent.run` invocation."""

    messages: list[Message] = field(default_factory=list)
    llm_calls: int = 0
    usage: Usage = field(default_factory=Usage)

    @property
    def final(self) -> str:
        """The last assistant message that contains text and no pending tool calls."""
        for message in reversed(self.messages):
            if isinstance(message, AssistantMessage) and not message.tool_calls:
                if message.content and message.content.strip():
                    return message.content
        return ""


class Agent:
    """A minimal ReAct agent that calls the OpenAI SDK directly."""

    def __init__(
        self,
        client,
        *,
        model_name: str | None = None,
        tools=None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_iterations: int = MAX_ITERATIONS,
        use_memory: bool = True,
        max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
    ):
        self.client = client
        self.model_name = model_name or settings.model_name
        self.tools = list(tools) if tools is not None else list(CODING_TOOLS)
        self.tools_by_name = {tool.name: tool for tool in self.tools}
        self.temperature = settings.temperature if temperature is None else temperature
        self.max_tokens = settings.max_tokens if max_tokens is None else max_tokens
        self.max_iterations = max_iterations
        self.use_memory = use_memory
        self.max_context_tokens = max_context_tokens
        self.store = get_store() if use_memory else None
        self.conversation_id = None

    def run(self, messages: list[Message], on_event=None, workspace: str | None = None) -> AgentResult:
        """Execute the agent loop.

        ``messages`` is the conversation history so far (typically a user
        message, or a longer thread). The agent appends a fresh system prompt on
        every model call. Only the messages it produces during this call are
        returned in ``result.messages`` and can be extended onto ``messages``.

        ``on_event`` receives :class:`~my_agent.events.AgentEvent` objects
        as the loop progresses (handy for streaming UIs).
        """
        conversation: list[Message] = list(messages)
        result = AgentResult()
        emit = on_event or (lambda _event: None)

        # Load persistent memory if enabled
        if self.use_memory and self.store:
            if self.conversation_id is None:
                self.conversation_id = self.store.get_or_create_conversation(workspace)
            # Load previous messages from memory
            stored_messages, summary = self.store.load_messages(
                self.conversation_id, self.max_context_tokens
            )
            # Merge: put stored history before current messages
            if stored_messages:
                conversation = stored_messages + conversation

        # Save new user messages to memory
        if self.use_memory and self.store and self.conversation_id:
            for msg in messages:
                from my_agent.messages import UserMessage

                if isinstance(msg, UserMessage):
                    self.store.save_message(self.conversation_id, msg)

        while result.llm_calls < self.max_iterations:
            # Check if we need to summarize before making the call
            if self.use_memory and self.store:
                self._maybe_summarize(conversation)

            response = self._call_model(conversation)
            result.llm_calls += 1
            self._accumulate_usage(result.usage, response)

            choice_message = response.choices[0].message
            assistant = from_openai(choice_message)
            conversation.append(assistant)
            result.messages.append(assistant)

            # Save assistant message to memory
            if self.use_memory and self.store and self.conversation_id:
                self.store.save_message(self.conversation_id, assistant)

            if assistant.content and assistant.content.strip():
                emit(AssistantEvent(text=assistant.content))

            if not assistant.tool_calls:
                break

            for call in assistant.tool_calls:
                emit(ToolCallEvent(id=call.id, name=call.name, arguments=call.arguments))
                tool_message = self._execute_call(call)
                conversation.append(tool_message)
                result.messages.append(tool_message)

                # Save tool message to memory
                if self.use_memory and self.store and self.conversation_id:
                    self.store.save_message(self.conversation_id, tool_message)

                emit(ToolResultEvent(id=tool_message.tool_call_id, name=call.name, content=tool_message.content))

        return result

    def _call_model(self, conversation: list[Message]):
        request_messages = [SystemMessage(content=build_system_prompt())] + list(conversation)
        openai_messages = [to_openai(message) for message in request_messages]
        kwargs: dict = {
            "model": self.model_name,
            "messages": openai_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.tools:
            kwargs["tools"] = [tool.to_openai() for tool in self.tools]
            kwargs["tool_choice"] = "auto"
        return self.client.chat.completions.create(**kwargs)

    def _execute_call(self, call: ToolCall) -> ToolMessage:
        if call.error is not None:
            content = f"Error: could not parse arguments for '{call.name}': {call.error}"
            return ToolMessage(tool_call_id=call.id, content=content, name=call.name)

        tool = self.tools_by_name.get(call.name)
        if tool is None:
            content = f"Error: unknown tool '{call.name}'."
            return ToolMessage(tool_call_id=call.id, content=content, name=call.name)

        try:
            output = tool.invoke(call.arguments)
        except Exception as exc:  # noqa: BLE001 - keep the agent alive on tool failures
            output = f"Error while running '{call.name}': {exc}"
        return ToolMessage(tool_call_id=call.id, content=str(output), name=call.name)

    def _maybe_summarize(self, conversation: list[Message]):
        """Summarize old messages if approaching token limit."""
        if not self.store or not self.conversation_id:
            return

        total_tokens = sum(count_message_tokens(m) for m in conversation)
        if total_tokens > int(self.max_context_tokens * 0.7) and len(conversation) > 15:
            # Keep last 10 messages, summarize the rest
            to_summarize = conversation[:-10]
            # Create a summary
            summary_parts = []
            for msg in to_summarize:
                if isinstance(msg, UserMessage):
                    summary_parts.append(f"User: {msg.content[:200]}")
                elif isinstance(msg, AssistantMessage):
                    if msg.content:
                        summary_parts.append(f"Assistant: {msg.content[:200]}")
                    if msg.tool_calls:
                        summary_parts.append(f"Assistant called: {', '.join(tc.name for tc in msg.tool_calls)}")
                elif isinstance(msg, ToolMessage):
                    summary_parts.append(f"Tool ({msg.name}): {msg.content[:100]}")
            
            summary = "\n".join(summary_parts)
            self.store.save_summary(self.conversation_id, summary)
            self.store.delete_old_messages(self.conversation_id, 10)

    @staticmethod
    def _accumulate_usage(usage: Usage, response) -> None:
        raw_usage = getattr(response, "usage", None)
        if raw_usage is None:
            return
        usage.add(
            prompt_tokens=getattr(raw_usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(raw_usage, "completion_tokens", 0) or 0,
            total_tokens=getattr(raw_usage, "total_tokens", 0) or 0,
        )

    def _execute_call(self, call: ToolCall) -> ToolMessage:
        if call.error is not None:
            content = f"Error: could not parse arguments for '{call.name}': {call.error}"
            return ToolMessage(tool_call_id=call.id, content=content, name=call.name)

        tool = self.tools_by_name.get(call.name)
        if tool is None:
            content = f"Error: unknown tool '{call.name}'."
            return ToolMessage(tool_call_id=call.id, content=content, name=call.name)

        try:
            output = tool.invoke(call.arguments)
        except Exception as exc:  # noqa: BLE001 - keep the agent alive on tool failures
            output = f"Error while running '{call.name}': {exc}"
        return ToolMessage(tool_call_id=call.id, content=str(output), name=call.name)

    @staticmethod
    def _accumulate_usage(usage: Usage, response) -> None:
        raw_usage = getattr(response, "usage", None)
        if raw_usage is None:
            return
        usage.add(
            prompt_tokens=getattr(raw_usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(raw_usage, "completion_tokens", 0) or 0,
            total_tokens=getattr(raw_usage, "total_tokens", 0) or 0,
        )


def build_agent(client=None, **kwargs) -> Agent:
    """Build an Agent. If no client is provided, one is created from settings."""
    if client is None:
        from my_agent.models.provider import get_client

        client = get_client()
    return Agent(client, **kwargs)


def run(agent, task: str) -> AgentResult:
    """Run a single task through an agent and return the result."""
    return agent.run([UserMessage(content=task)])
