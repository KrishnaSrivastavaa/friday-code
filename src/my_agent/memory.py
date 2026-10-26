"""Persistent conversation storage with SQLite.

Stores conversations per workspace (folder) like OpenCode. Handles token-aware
history loading with automatic summarization of old messages when approaching
context limits.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tiktoken

from src.my_agent.config.settings import settings
from src.my_agent.messages import (
    AssistantMessage,
    Message,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from src.my_agent.tools._common import workspace_root


# Default max tokens before summarization (conservative for gpt-4o)
DEFAULT_MAX_CONTEXT_TOKENS = 120000  # Leave room for response
SUMMARY_TRIGGER_RATIO = 0.7  # Trigger summarization at 70% of max
KEEP_RECENT_MESSAGES = 10  # Keep last N messages unsummarized


def get_storage_path() -> Path:
    """Get the path to the SQLite database."""
    home = Path.home()
    storage_dir = home / ".pypilot"
    storage_dir.mkdir(parents=True, exist_ok=True)
    return storage_dir / "memory.db"


def get_encoder():
    """Get tiktoken encoder for token counting."""
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def count_tokens(text: str) -> int:
    """Count tokens in text."""
    encoder = get_encoder()
    if encoder is None:
        # Fallback rough estimate
        return len(text) // 4
    try:
        return len(encoder.encode(text))
    except Exception:
        return len(text) // 4


def count_message_tokens(message: Message) -> int:
    """Count approximate tokens in a message."""
    tokens = 0
    if isinstance(message, UserMessage):
        tokens += count_tokens(message.content)
        tokens += 4  # Message overhead
    elif isinstance(message, AssistantMessage):
        tokens += count_tokens(message.content)
        if message.tool_calls:
            for call in message.tool_calls:
                tokens += count_tokens(call.name)
                tokens += count_tokens(json.dumps(call.arguments))
        tokens += 4
    elif isinstance(message, ToolMessage):
        tokens += count_tokens(message.content)
        tokens += 4
    return tokens


class MemoryStore:
    """SQLite-backed persistent memory store."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or get_storage_path()
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        conn = sqlite3.connect(str(self.db_path))
        cur = conn.cursor()
        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                workspace_path TEXT NOT NULL,
                title TEXT,
                summary TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                total_tokens INTEGER DEFAULT 0
            )
        """)
        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                tool_call_id TEXT,
                tool_name TEXT,
                tool_arguments TEXT,
                tool_calls_json TEXT,
                created_at REAL NOT NULL,
                token_count INTEGER DEFAULT 0,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            )
        """)
        
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_conversation 
            ON messages(conversation_id, created_at)
        """)
        
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversations_workspace 
            ON conversations(workspace_path, updated_at)
        """)
        
        conn.commit()
        conn.close()

    def get_or_create_conversation(self, workspace: str | Path | None = None) -> str:
        """Get the most recent conversation for a workspace, or create a new one."""
        if workspace is None:
            workspace = str(workspace_root())
        workspace = str(Path(workspace).resolve())
        
        conn = sqlite3.connect(str(self.db_path))
        cur = conn.cursor()
        
        # Find most recent conversation for this workspace
        cur.execute(
            """
            SELECT id FROM conversations 
            WHERE workspace_path = ? 
            ORDER BY updated_at DESC 
            LIMIT 1
            """,
            (workspace,),
        )
        row = cur.fetchone()
        
        if row:
            conv_id = row[0]
            conn.close()
            return conv_id
        
        # Create new conversation
        conv_id = str(uuid.uuid4())
        now = time.time()
        cur.execute(
            """
            INSERT INTO conversations (id, workspace_path, title, summary, created_at, updated_at, total_tokens)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (conv_id, workspace, f"Conversation {time.strftime('%Y-%m-%d %H:%M')}", None, now, now, 0),
        )
        conn.commit()
        conn.close()
        return conv_id

    def save_message(self, conversation_id: str, message: Message):
        """Save a message to the database."""
        conn = sqlite3.connect(str(self.db_path))
        cur = conn.cursor()
        
        msg_id = str(uuid.uuid4())
        now = time.time()
        token_count = count_message_tokens(message)
        
        if isinstance(message, UserMessage):
            cur.execute(
                """
                INSERT INTO messages (id, conversation_id, role, content, created_at, token_count)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (msg_id, conversation_id, "user", message.content, now, token_count),
            )
        elif isinstance(message, AssistantMessage):
            tool_calls_json = None
            if message.tool_calls:
                tool_calls_json = json.dumps([
                    {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                    for tc in message.tool_calls
                ])
            cur.execute(
                """
                INSERT INTO messages (id, conversation_id, role, content, tool_calls_json, created_at, token_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (msg_id, conversation_id, "assistant", message.content, tool_calls_json, now, token_count),
            )
        elif isinstance(message, ToolMessage):
            cur.execute(
                """
                INSERT INTO messages (id, conversation_id, role, content, tool_call_id, tool_name, created_at, token_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (msg_id, conversation_id, "tool", message.content, message.tool_call_id, message.name, now, token_count),
            )
        
        # Update conversation timestamp and token count
        cur.execute(
            """
            UPDATE conversations 
            SET updated_at = ?, total_tokens = total_tokens + ?
            WHERE id = ?
            """,
            (now, token_count, conversation_id),
        )
        
        conn.commit()
        conn.close()

    def load_messages(self, conversation_id: str, max_tokens: int | None = None) -> tuple[list[Message], str | None]:
        """Load messages for a conversation, summarizing if over token limit."""
        if max_tokens is None:
            max_tokens = DEFAULT_MAX_CONTEXT_TOKENS
        
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        
        # Get conversation summary
        cur.execute("SELECT summary FROM conversations WHERE id = ?", (conversation_id,))
        row = cur.fetchone()
        summary = row["summary"] if row and row["summary"] else None
        
        # Get all messages
        cur.execute(
            """
            SELECT * FROM messages 
            WHERE conversation_id = ? 
            ORDER BY created_at ASC
            """,
            (conversation_id,),
        )
        rows = cur.fetchall()
        conn.close()
        
        messages: list[Message] = []
        total_tokens = count_tokens(summary) if summary else 0
        
        for row in rows:
            msg = self._row_to_message(row)
            total_tokens += count_message_tokens(msg)
            messages.append(msg)
        
        # If over limit, we need to summarize
        if total_tokens > int(max_tokens * SUMMARY_TRIGGER_RATIO) and len(messages) > KEEP_RECENT_MESSAGES:
            # Keep recent messages, summarize the rest
            to_summarize = messages[:-KEEP_RECENT_MESSAGES]
            recent = messages[-KEEP_RECENT_MESSAGES:]
            
            # Generate summary (this will be called by the agent if needed)
            # For now, return recent messages with a note - actual summarization happens elsewhere
            messages = recent
        
        return messages, summary

    def _row_to_message(self, row) -> Message:
        """Convert database row to Message object."""
        role = row["role"]
        if role == "user":
            return UserMessage(content=row["content"] or "")
        elif role == "assistant":
            tool_calls = []
            if row["tool_calls_json"]:
                try:
                    calls_data = json.loads(row["tool_calls_json"])
                    for cd in calls_data:
                        tool_calls.append(ToolCall(id=cd["id"], name=cd["name"], arguments=cd["arguments"]))
                except Exception:
                    pass
            return AssistantMessage(content=row["content"] or "", tool_calls=tool_calls)
        elif role == "tool":
            return ToolMessage(
                tool_call_id=row["tool_call_id"] or "",
                content=row["content"] or "",
                name=row["tool_name"],
            )
        return UserMessage(content="")

    def save_summary(self, conversation_id: str, summary: str):
        """Save a summary for a conversation."""
        conn = sqlite3.connect(str(self.db_path))
        cur = conn.cursor()
        cur.execute(
            "UPDATE conversations SET summary = ? WHERE id = ?",
            (summary, conversation_id),
        )
        conn.commit()
        conn.close()

    def delete_old_messages(self, conversation_id: str, keep_count: int = KEEP_RECENT_MESSAGES):
        """Delete old messages after summarization, keeping only recent ones."""
        conn = sqlite3.connect(str(self.db_path))
        cur = conn.cursor()
        
        # Get messages to delete (all except most recent)
        cur.execute(
            """
            SELECT id FROM messages 
            WHERE conversation_id = ? 
            ORDER BY created_at ASC
            LIMIT -1 OFFSET ?
            """,
            (conversation_id, keep_count),
        )
        to_delete = [row[0] for row in cur.fetchall()]
        
        if to_delete:
            cur.execute(
                f"DELETE FROM messages WHERE id IN ({','.join('?' * len(to_delete))})",
                to_delete,
            )
        
        conn.commit()
        conn.close()

    def list_conversations(self, workspace: str | Path | None = None) -> list[dict]:
        """List all conversations for a workspace."""
        if workspace is None:
            workspace = str(workspace_root())
        workspace = str(Path(workspace).resolve())
        
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        
        cur.execute(
            """
            SELECT id, title, created_at, updated_at, total_tokens 
            FROM conversations 
            WHERE workspace_path = ? 
            ORDER BY updated_at DESC
            """,
            (workspace,),
        )
        rows = cur.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]


# Global instance
_store = None


def get_store() -> MemoryStore:
    """Get the global memory store instance."""
    global _store
    if _store is None:
        _store = MemoryStore()
    return _store
