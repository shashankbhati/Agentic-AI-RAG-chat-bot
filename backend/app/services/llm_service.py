import logging
from typing import List, Dict, Generator
from app.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a precise document assistant. "
    "Answer questions strictly using the provided context. "
    "If the answer is not found in the context, say: "
    "'I couldn't find this information in the provided documents.' "
    "Be concise and accurate."
)


def _build_prompt(query: str, context: str, history: List[Dict]) -> str:
    history_text = ""
    if history:
        for msg in history[-4:]:  # last 2 turns
            role = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n"

    parts = [f"Context from documents:\n{context}"]
    if history_text:
        parts.append(f"Conversation history:\n{history_text.strip()}")
    parts.append(f"Question: {query}\n\nAnswer:")

    return "\n\n".join(parts)


class LLMService:
    def generate(self, query: str, context: str, history: List[Dict] = None) -> str:
        prompt = _build_prompt(query, context, history or [])
        if settings.LLM_PROVIDER == "openai":
            return self._openai_generate(prompt)
        return self._ollama_generate(prompt)

    def stream(self, query: str, context: str, history: List[Dict] = None) -> Generator[str, None, None]:
        prompt = _build_prompt(query, context, history or [])
        if settings.LLM_PROVIDER == "openai":
            yield from self._openai_stream(prompt)
        else:
            yield from self._ollama_stream(prompt)

    def _ollama_generate(self, prompt: str) -> str:
        import ollama
        response = ollama.chat(
            model=settings.OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0},
        )
        return response["message"]["content"]

    def _ollama_stream(self, prompt: str) -> Generator[str, None, None]:
        import ollama
        stream = ollama.chat(
            model=settings.OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0},
            stream=True,
        )
        for chunk in stream:
            content = chunk.get("message", {}).get("content", "")
            if content:
                yield content

    def _openai_generate(self, prompt: str) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        return response.choices[0].message.content

    def _openai_stream(self, prompt: str) -> Generator[str, None, None]:
        from openai import OpenAI
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        stream = client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            stream=True,
        )
        for chunk in stream:
            content = chunk.choices[0].delta.content or ""
            if content:
                yield content

    def health_check(self) -> bool:
        if settings.LLM_PROVIDER == "openai":
            return bool(settings.OPENAI_API_KEY)
        try:
            import ollama
            ollama.list()
            return True
        except Exception:
            return False


llm_service = LLMService()
