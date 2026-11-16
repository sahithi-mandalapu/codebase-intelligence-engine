from __future__ import annotations

import logging
import os

log = logging.getLogger("cie.llm")

SYSTEM = (
    "You answer questions about a codebase using only the snippets provided. "
    "Cite snippets inline like [1] or [2][3]. If the snippets don't contain the answer, say so "
    "instead of guessing. Mention file paths and function names where you can. "
    "Keep answers short and concrete."
)


def build_prompt(question: str, hits, notes: str = "") -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        c = h.chunk
        blocks.append(f"[{i}] {c.header()}\n```{c.language}\n{c.text}\n```")
    prompt = f"Question: {question}\n\n"
    if notes:
        prompt += f"Structure notes (from static analysis of the repo):\n{notes}\n\n"
    prompt += "Snippets:\n\n" + "\n\n".join(blocks)
    return prompt


class ExtractiveLLM:
    """no api key needed. just lists what retrieval found, so the demo works offline."""

    name = "none"

    def answer(self, question: str, hits, notes: str = "") -> str:
        lines = ["(no LLM configured, so this is just what retrieval found)"]
        if notes:
            lines += ["", notes]
        lines.append("")
        for i, h in enumerate(hits, 1):
            lines.append(f"[{i}] {h.chunk.header()}")
        return "\n".join(lines)


class AnthropicLLM:
    name = "anthropic"

    def __init__(self, model: str | None = None):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model or os.getenv("CIE_ANTHROPIC_MODEL", "claude-sonnet-5-5")

    def answer(self, question: str, hits, notes: str = "") -> str:
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM,
            messages=[{"role": "user", "content": build_prompt(question, hits, notes)}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


class OpenAILLM:
    name = "openai"

    def __init__(self, model: str | None = None):
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model or os.getenv("CIE_OPENAI_MODEL", "gpt-4o-mini")

    def answer(self, question: str, hits, notes: str = "") -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": build_prompt(question, hits, notes)},
            ],
            max_tokens=1024,
        )
        return resp.choices[0].message.content or ""


def get_llm(name: str = "none"):
    try:
        if name == "anthropic":
            return AnthropicLLM()
        if name == "openai":
            return OpenAILLM()
    except Exception as e:  # missing package or api key
        log.warning("couldn't start %s llm (%s), falling back to extractive answers", name, e)
    return ExtractiveLLM()
