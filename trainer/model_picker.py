"""Proposes which local and Compass models to benchmark, with a built-in local fallback."""
from __future__ import annotations

import json
import os
import re
from typing import Any

from compass_client import CompassError, chat

MAX_LOCAL = 4
MAX_COMPASS = 3
INSTRUCTION = (
    "You pick models for a benchmark before fine-tuning. From the offered local and Compass models, choose the ones worth comparing for this task. "
    'Return JSON only: {"recommendation": str, "models": [{"id": str, "provider": "local"|"compass", "reason": str}], "excluded": [{"id": str, "reason": str}]}. '
    f"Use only offered ids; at most {MAX_LOCAL} local and {MAX_COMPASS} compass models; give a short reason for each."
)


def eligible(local: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in local if not item.get("rejected")]


def built_in_proposal(local: list[dict[str, Any]], note: str) -> dict[str, Any]:
    picked = eligible(local)[:MAX_LOCAL]
    models = [{"id": item["model"]["id"], "provider": "local", "reason": (item.get("reasons") or ["Fits the GPU"])[0]} for item in picked]
    excluded = [{"id": item["model"]["id"], "reason": item["rejected"]} for item in local if item.get("rejected")]
    return {"source": "built_in", "note": note, "recommendation": f"Start with {models[0]['id']}." if models else "", "models": models, "excluded": excluded}


def plain_facts(facts: Any) -> dict[str, Any]:
    plain = (str, int, float, bool)
    kept = {}
    for key, value in (facts if isinstance(facts, dict) else {}).items():
        if isinstance(value, plain) or (isinstance(value, dict) and all(isinstance(inner, plain) for inner in value.values())):
            kept[key] = value
    return kept


def build_prompt(brief: dict[str, Any], profile: dict[str, Any], memory_gb: int, local: list[dict[str, Any]], compass: list[str]) -> str:
    classification = profile.get("classification")
    classification = {key: classification.get(key) for key in ("task", "confidence", "source")} if isinstance(classification, dict) else None
    models = [{"id": item["model"]["id"], **{key: item["model"].get(key) for key in ("sizeB", "vram", "task")}} for item in eligible(local)]
    return json.dumps({
        "task": {key: brief.get(key) for key in ("task_description", "success_metric", "analysis_limit")},
        "classification": classification,
        "facts": plain_facts(profile.get("facts")),
        "gpuMemoryGb": memory_gb,
        "localModels": models,
        "compassModels": compass,
    })


def parse_reply(text: str) -> dict[str, Any]:
    data = json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip()))
    if not isinstance(data, dict):
        raise ValueError("reply is not a JSON object")
    return data


def valid_entries(items: Any, allowed: dict[str, str], keep_provider: bool) -> list[dict[str, str]]:
    seen, result = set(), []
    for item in items if isinstance(items, list) else []:
        model_id = item.get("id") if isinstance(item, dict) else None
        if isinstance(model_id, str) and model_id in allowed and model_id not in seen:
            seen.add(model_id)
            entry = {"id": model_id, "reason": str(item.get("reason") or "")}
            result.append({**entry, "provider": allowed[model_id]} if keep_provider else entry)
    return result


def propose_models(brief: dict[str, Any], profile: dict[str, Any], memory_gb: int, local: list[dict[str, Any]], compass: list[str]) -> dict[str, Any]:
    allowed = {**{model_id: "compass" for model_id in compass}, **{item["model"]["id"]: "local" for item in eligible(local)}}
    messages = [{"role": "system", "content": INSTRUCTION}, {"role": "user", "content": build_prompt(brief, profile, memory_gb, local, compass)}]
    try:
        data = parse_reply(chat(os.environ.get("ADVISOR_MODEL") or "gpt-4.1-mini", messages, max_tokens=1500)["output"])
        models = valid_entries(data.get("models"), allowed, True)
        picked = [m for m in models if m["provider"] == "local"][:MAX_LOCAL] + [m for m in models if m["provider"] == "compass"][:MAX_COMPASS]
        if not picked:
            raise ValueError("no valid models in the reply")
    except (CompassError, ValueError) as error:
        return built_in_proposal(local, f"AI picker unavailable: {error}; showing the built-in local ranking.")
    return {"source": "agent", "note": "", "recommendation": str(data.get("recommendation") or ""), "models": picked, "excluded": valid_entries(data.get("excluded"), allowed, False)}
