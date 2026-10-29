import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any, Callable

ENDPOINT = "https://compass.llm.shopee.io/compass-api/v1"
NON_TEXT_MODELS = re.compile(r"chirp|nmt|byteplus|-vl-|tts|whisper|embed|image|video|dall-e|imagen|veo|sora|audio", re.IGNORECASE)


class CompassError(Exception):
    pass


def configured() -> bool:
    return bool(os.getenv("ADVISOR_API_KEY")) and os.getenv("ADVISOR_BASE_URL", "").rstrip("/") == ENDPOINT


def safe_message(message: Any) -> str:
    key = os.getenv("ADVISOR_API_KEY")
    text = str(message or "Compass request failed")
    return (text.replace(key, "[redacted]") if key else text)[:500]


def request_compass(path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    if not configured():
        raise CompassError("Set ADVISOR_API_KEY and the Compass ADVISOR_BASE_URL in the trainer's .env, then restart the trainer.")
    headers = {"Authorization": "Bearer " + os.environ["ADVISOR_API_KEY"], "Content-Type": "application/json"}
    for attempt in range(3):
        request = urllib.request.Request(ENDPOINT + path, data=json.dumps(body).encode() if body is not None else None, headers=headers, method="POST" if body is not None else "GET")
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            if error.code == 429 and attempt < 2:
                time.sleep(30)
                continue
            try:
                message = json.loads(error.read()).get("message")
            except (ValueError, AttributeError, OSError):
                message = None
            raise CompassError(safe_message(message or f"Compass returned HTTP {error.code}.")) from None
        except (urllib.error.URLError, OSError, ValueError) as error:
            raise CompassError(safe_message(error)) from None
    raise CompassError("Compass request failed")


def text_models() -> list[str]:
    ids = (item.get("id") for item in request_compass("/models").get("data", []))
    return sorted(model for model in ids if isinstance(model, str) and not NON_TEXT_MODELS.search(model))


def chat(model: str, messages: list[dict[str, Any]], max_tokens: int = 256, temperature: float = 0) -> dict[str, Any]:
    kept = [{"role": m["role"], "content": m["content"]} for m in messages if not (m["role"] == "system" and not str(m["content"]).strip())]
    if not kept or not str(kept[-1]["content"]).strip():
        raise CompassError("Conversation input is empty.")
    started = time.monotonic()
    data = request_compass("/chat/completions", {"model": model, "messages": kept, "temperature": temperature, "max_tokens": max_tokens})
    choice = (data.get("choices") or [{}])[0]
    output = (choice.get("message") or {}).get("content")
    if not isinstance(output, str) or not output.strip():
        raise CompassError("Compass returned no answer text.")
    return {"output": output, "finishReason": choice.get("finish_reason"), "usage": data.get("usage") or {}, "seconds": time.monotonic() - started}


def answer_cases(model: str, cases: list[dict[str, Any]], on_answer: Callable[[int, int], None] | None = None) -> list[dict[str, Any]]:
    results = []
    for case in cases:
        answer = chat(model, case["messages"])
        results.append({"model": model, "rowId": case["rowId"], **answer})
        if on_answer:
            on_answer(len(results), len(cases))
    return results
