"""Stream revision-pinned Hub samples or complete splits to local storage."""
import json
import re
import shutil
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit
from datasets import get_dataset_config_names, load_dataset
from huggingface_hub import dataset_info

SAMPLE_BYTES = 50 * 1024 * 1024
DISK_RESERVE = 1024 * 1024 * 1024


def dataset_id(value):
    value = value.strip()
    if value.startswith("https://"):
        url = urlsplit(value)
        if url.netloc != "huggingface.co" or not url.path.startswith("/datasets/"):
            raise ValueError("Use a Hugging Face dataset URL or owner/dataset ID.")
        value = url.path[len("/datasets/"):].rstrip("/")
    if not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*/[A-Za-z0-9_-][A-Za-z0-9_.-]*", value) or ".." in value:
        raise ValueError("Use a dataset page URL or owner/dataset ID, not a file or viewer URL.")
    return value


def import_huggingface(value, configuration, split, max_rows, directory, *, progress=None, cancelled=lambda: False, import_id=None, resolved=None):
    repo = dataset_id(value)
    if (max_rows is not None and (isinstance(max_rows, bool) or not isinstance(max_rows, int) or not 30 <= max_rows <= 10000)) or not split or len(split) > 100:
        raise ValueError("Choose 30–10000 rows and a valid split.")
    def check_cancelled():
        if cancelled():
            raise InterruptedError("Import cancelled. No partial dataset was kept.")
    check_cancelled()
    revision = resolved[0] if resolved else dataset_info(repo).sha
    configs = resolved[1] if resolved else get_dataset_config_names(repo, revision=revision)
    check_cancelled()
    if configuration is None and len(configs) > 1:
        return {"requiresConfiguration": True, "configurations": configs, "datasetId": repo}
    configuration = configuration or (configs[0] if configs else None)
    if configs and configuration not in configs:
        raise ValueError("Unknown dataset subset. Choose one of: " + ", ".join(configs))
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(directory).free < DISK_RESERVE:
        raise ValueError("Not enough disk space on the trainer. Keep at least 1 GiB free and space for the dataset cache.")
    stream = load_dataset(repo, name=configuration, split=split, revision=revision, streaming=True)
    destination = directory / ((str(uuid.UUID(import_id)) if import_id else str(uuid.uuid4())) + ".jsonl")
    partial = destination.with_suffix(".jsonl.partial")
    if destination.exists() or partial.exists():
        raise ValueError("This import already has a dataset file.")
    splits = getattr(getattr(stream, "info", None), "splits", None) or {}
    total_rows = getattr(splits.get(split), "num_examples", None)
    if total_rows is not None and max_rows is not None:
        total_rows = min(total_rows, max_rows)
    total = count = 0
    last_update = disk_budget = 0
    try:
        with partial.open("xb") as output:
            # ponytail: sample mode uses a prefix; use stratified sampling for representative evaluation.
            for row in stream.take(max_rows) if max_rows is not None else stream:
                check_cancelled()
                line = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
                if max_rows is not None and total + len(line) > SAMPLE_BYTES:
                    raise ValueError("The sample exceeds 50 MB. Import fewer rows or a smaller subset.")
                # Reserve estimated Arrow-cache space too; periodically account for other disk users.
                if count % 100 == 0 or disk_budget < 2 * len(line):
                    disk_budget = shutil.disk_usage(directory).free - DISK_RESERVE - total
                if disk_budget < 2 * len(line):
                    raise ValueError("Not enough disk space on the trainer for the import and dataset cache. No partial dataset was kept.")
                output.write(line)
                total += len(line)
                disk_budget -= 2 * len(line)
                count += 1
                if progress and time.monotonic() - last_update >= 1:
                    progress({"importedRows": count, "importedBytes": total, "totalRows": total_rows})
                    last_update = time.monotonic()
        if not count:
            raise ValueError("The selected subset/split has no rows.")
        check_cancelled()
        if progress:
            progress({"importedRows": count, "importedBytes": total, "totalRows": total_rows})
        partial.replace(destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    scope = "sample" if max_rows is not None else "full split"
    return {"source":"upload", "name":str(destination), "extension":"jsonl", "displayName":f"{repo} · {configuration or 'default'} · {split} · {scope}"[:255], "sourceUrl":"https://huggingface.co/datasets/" + repo, "importedRows":count, "hubConfig":configuration, "hubSplit":split, "hubRevision":revision, "sampleOnly":max_rows is not None}


def import_all_subsets(value, split, directory, *, progress, cancelled, import_id):
    """Keep each configuration separate and audit all of them before selection."""
    from advisor import load_source, local_profile
    if cancelled():
        raise InterruptedError("Import cancelled.")
    repo = dataset_id(value)
    revision = dataset_info(repo).sha
    configs = get_dataset_config_names(repo, revision=revision) or [None]
    if cancelled():
        raise InterruptedError("Import cancelled.")
    root = Path(directory) / str(uuid.UUID(import_id))
    root.mkdir(parents=True, exist_ok=True)
    subsets = []
    rows = size = 0
    for index, config in enumerate(configs):
        if cancelled():
            raise InterruptedError("Import cancelled. Completed subsets are retained; unfinished downloads are removed.")
        entry = {"configuration": config, "status":"importing"}
        subsets.append(entry)
        current = {"importedRows":0, "importedBytes":0}
        def report(values):
            current.update(values)
            progress({"phase":entry["status"], "currentSubset":config or "default",
                      "completedSubsets":index, "totalSubsets":len(configs), "subsets":subsets,
                      "importedRows":rows + current["importedRows"], "importedBytes":size + current["importedBytes"], "totalRows":None})
        report({})
        try:
            child_id = str(uuid.uuid5(uuid.UUID(import_id), json.dumps(config)))
            ref = import_huggingface(repo, config, split, None, root, progress=report,
                                    cancelled=cancelled, import_id=child_id, resolved=(revision, configs))
            entry.update(dataset=ref, status="analyzing")
            report({})
            if cancelled():
                raise InterruptedError("Import cancelled. Completed subsets are retained.")
            dataset = load_source(ref["source"], ref["name"], ref["extension"])
            entry.update(profile=local_profile(dataset), status="analyzed")
        except InterruptedError:
            raise
        except Exception as error:
            entry.update(status="failed", error=str(error)[:500])
        rows += entry.get("dataset", {}).get("importedRows", 0)
        size += Path(entry["dataset"]["name"]).stat().st_size if entry.get("dataset") else 0
        progress({"phase":"analyzing", "completedSubsets":index + 1, "totalSubsets":len(configs),
                  "currentSubset":config or "default", "subsets":subsets, "importedRows":rows, "importedBytes":size, "totalRows":None})
    if cancelled():
        raise InterruptedError("Import cancelled. Completed subsets are retained.")
    return {"allSubsets":True, "datasetId":repo, "hubRevision":revision, "hubSplit":split,
            "subsets":subsets, "importedRows":rows, "failedSubsets":sum(item["status"] == "failed" for item in subsets)}
