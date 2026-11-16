"""
LLM 呼叫模組：使用 OpenAI 相容 API，所以同一份程式可以接
- OpenAI（gpt-4o-mini 等）
- Google Gemini（OpenAI 相容端點）
- 本機 Ollama（http://localhost:11434/v1，免費、資料不外流）
沒有設定時自動進入「示範模式」，讓整個 Agent 流程仍可跑通。
"""
import json
import os
import re

BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("LLM_API_KEY", "")
MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

MOCK = not API_KEY and "localhost" not in BASE_URL and "127.0.0.1" not in BASE_URL


def chat(system: str, user: str, temperature: float = 0.7) -> str:
    if MOCK:
        raise RuntimeError("mock")
    import requests
    resp = requests.post(
        f"{BASE_URL.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY or 'ollama'}"},
        json={
            "model": MODEL,
            "temperature": temperature,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def chat_json(system: str, user: str) -> dict:
    """要求模型回傳 JSON，並寬鬆解析（模型有時會包 ```json）"""
    raw = chat(system + "\n只輸出 JSON，不要任何其他文字。", user, temperature=0.2)
    m = re.search(r"\{.*\}", raw, re.S)
    return json.loads(m.group(0) if m else raw)
