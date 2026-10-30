"""A small per-model history of benchmark results, so the model picker learns between runs.

It stores model IDs, scores, costs and errors only; never questions, answers or dataset rows.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

MAX_ENTRIES = 500
MAX_SUMMARY = 15


def history_path(data_root: Path) -> Path:
    return Path(data_root) / "model-history.json"


def _load(data_root: Path) -> list[dict[str, Any]]:
    try:
        entries = json.loads(history_path(data_root).read_text(encoding="utf-8")).get("entries")
    except (OSError, ValueError, AttributeError):
        return []
    return [entry for entry in entries if isinstance(entry, dict)] if isinstance(entries, list) else []


def _save(data_root: Path, entries: list[dict[str, Any]]) -> None:
    path = history_path(data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp = tempfile.mkstemp(dir=path.parent, prefix=".model-history-", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as file:
            json.dump({"version": 1, "entries": entries[-MAX_ENTRIES:]}, file)
        os.replace(temp, path)
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise


def _number(value: Any) -> float | None:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def _date(when: datetime | None) -> str:
    return (when or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()


def _parse(date: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(date)
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _entry(row: dict[str, Any], task: Any, date: str, workflow_id: Any) -> dict[str, Any]:
    evidence = row.get("qualityEvidence") if isinstance(row.get("qualityEvidence"), dict) else {}
    status = "complete" if row.get("status") == "complete" else "unavailable"
    provider = "compass" if row.get("provider") == "compass" else "local"
    error = row.get("message")
    return {
        "modelId": str(row.get("modelId")), "provider": provider, "task": task, "date": date, "status": status,
        "score": _number(evidence.get("score")) if evidence.get("coverage") == 1 else None, "metric": evidence.get("metric"),
        "costUsd": _number(row.get("costUsd")), "meanLatencySeconds": _number(row.get("meanLatencySeconds")),
        "questions": len(row.get("evaluationIds") or []), "error": str(error)[:300] if error else None,
        "definite": provider == "compass" and status == "unavailable" and bool(row.get("unavailableDefinite")),
        "source": "benchmark", "workflowId": workflow_id,
    }


def record_benchmark(data_root: Path, task: Any, results: list[dict[str, Any]], workflow_id: Any, when: datetime | None = None) -> None:
    date = _date(when)
    _save(data_root, _load(data_root) + [_entry(row, task, date, workflow_id) for row in results if isinstance(row, dict) and row.get("modelId")])


def record_check_failure(data_root: Path, task: Any, model_id: str, message: Any, workflow_id: Any, when: datetime | None = None) -> None:
    entry = _entry({"modelId": model_id, "provider": "compass", "status": "unavailable", "message": message, "unavailableDefinite": True}, task, _date(when), workflow_id)
    _save(data_root, _load(data_root) + [{**entry, "source": "check"}])


def recently_unavailable(data_root: Path, now: datetime | None = None, days: int = 7) -> dict[str, dict[str, Any]]:
    since = (now or datetime.now(timezone.utc)) - timedelta(days=days)
    latest: dict[str, tuple[datetime, dict[str, Any]]] = {}
    for entry in _load(data_root):
        date = _parse(entry.get("date"))
        # Temporary failures say nothing either way; only a success or a definite refusal changes the verdict.
        if entry.get("provider") != "compass" or date is None or (entry.get("status") != "complete" and not entry.get("definite")):
            continue
        if entry.get("modelId") not in latest or date >= latest[entry["modelId"]][0]:
            latest[entry["modelId"]] = (date, entry)
    return {model: {"date": date, "error": entry.get("error")} for model, (date, entry) in latest.items() if entry.get("definite") and date >= since}


def _mean(values: list[Any]) -> float | None:
    numbers = [value for value in map(_number, values) if value is not None]
    return round(sum(numbers) / len(numbers), 6) if numbers else None


def summary_for(data_root: Path, task: Any) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for entry in _load(data_root):
        if entry.get("task") == task and entry.get("modelId"):
            groups.setdefault(entry["modelId"], []).append(entry)
    rows = []
    for model, entries in groups.items():
        last = max(entries, key=lambda entry: _parse(entry.get("date")) or datetime.min.replace(tzinfo=timezone.utc))
        scores = [entry.get("score") for entry in entries if _number(entry.get("score")) is not None]
        rows.append({
            "modelId": model, "provider": last.get("provider"), "runs": len(entries), "scoredRuns": len(scores), "meanScore": _mean(scores),
            "meanCostUsd": _mean([entry.get("costUsd") for entry in entries]), "meanLatencySeconds": _mean([entry.get("meanLatencySeconds") for entry in entries]),
            "lastDate": last.get("date"), "lastStatus": last.get("status"),
        })
    rows.sort(key=lambda row: (row["meanScore"] is None, -(row["meanScore"] or 0)))
    return rows[:MAX_SUMMARY]


def backfill_if_empty(data_root: Path, workflows_root: Path) -> None:
    if history_path(data_root).exists():
        return
    entries = []
    for path in sorted(Path(workflows_root).glob("*/workflow.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            state = record.get("state") or {}
            results = (state.get("model_benchmark") or {}).get("results") or []
            task = ((state.get("profile") or {}).get("classification") or {}).get("task")
            date = _parse(record.get("createdAt"))
            entries += [_entry(row, task, _date(date), record.get("id") or path.parent.name) for row in results if isinstance(row, dict) and row.get("modelId")]
        except (OSError, ValueError, AttributeError, TypeError):
            continue
    entries.sort(key=lambda entry: entry["date"])
    _save(data_root, entries)
