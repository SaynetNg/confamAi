"""Local conversation memory and client-side context management.

The conversation history lives on the laptop. When it grows too large the
client trims old turns (keeping the first message and the newest messages)
instead of silently destroying the whole conversation.
"""

import math

from .storage import load_conversation_from_disk, save_conversation_to_disk


def estimate_tokens(text) -> int:
    """Cheap heuristic token counter (no tokenizer on the laptop)."""
    if not text:
        return 0
    count = 0
    for word in str(text).split():
        count += 1 + int(len(word) / 4.0)
    return count


class Conversation:
    """A simple in-memory list of {role, content} messages."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.messages: list[dict] = []

    def reset(self) -> None:
        self.messages.clear()

    def add(self, role: str, content: str) -> None:
        text = str(content or "").strip()
        if not text:
            return
        self.messages.append({"role": role, "content": text})

    def build_messages_for(self, user_text: str) -> list[dict]:
        """Return the message payload to send for the new user message.

        The Confam AI system prompt (prompt.py) goes first, then the local
        history, then the new message. The prompt is empty only when it was
        switched off with N_ATLAS_SYSTEM_PROMPT=none.
        History is trimmed to MAX_CONVERSATION_MESSAGES and to an estimated
        MAX_CONVERSATION_TOKENS, preserving the opening exchange and the most
        recent turns.
        """
        result: list[dict] = []
        if self.cfg.system_prompt:
            result.append({"role": "system", "content": self.cfg.system_prompt})
        result.extend(self.messages)
        result.append({"role": "user", "content": user_text})

        if len(result) > self.cfg.max_conversation_messages:
            keep_first = sum(1 for m in result if m["role"] == "system")
            system_head = result[:keep_first]
            tail_count = self.cfg.max_conversation_messages - len(system_head)
            result = system_head + result[-tail_count:]

        guard = 0
        while (
            self._token_estimate(result) > self.cfg.max_conversation_tokens
            and guard < 200
        ):
            drop_at = self._drop_index(result)
            if drop_at is None:
                break
            del result[drop_at]
            guard += 1
        return result

    @staticmethod
    def _token_estimate(messages: list[dict]) -> int:
        return sum(estimate_tokens(m["content"]) for m in messages)

    @staticmethod
    def _drop_index(messages: list[dict]):
        """Index of the oldest removable user turn (preferred as a user +
        assistant pair), never the final message."""
        if not messages:
            return None
        last = len(messages) - 1
        for index, message in enumerate(messages):
            if message["role"] == "system" or index == last:
                continue
            if message["role"] == "user":
                if (
                    index + 1 <= last
                    and messages[index + 1]["role"] == "assistant"
                ):
                    return index
                return index
        return None

    def to_dict(self) -> list[dict]:
        return [dict(message) for message in self.messages]

    def token_estimate(self) -> int:
        return self._token_estimate(self.messages)

    # ------------------------------------------------------------------
    # storage convenience
    # ------------------------------------------------------------------
    def save(self, cfg, directory=None):
        if not self.messages:
            return None
        return save_conversation_to_disk(cfg, self.messages, directory=directory)

    def load(self, path) -> int:
        messages, _model = load_conversation_from_disk(path)
        self.messages = [dict(m) for m in messages]
        return len(self.messages)


def trim_to_capacity(messages: list[dict], cfg) -> list[dict]:
    """Utility used by tests / future callers: standalone trimming."""
    return Conversation(cfg).build_messages_for("")[:-1] or messages