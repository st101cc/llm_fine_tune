"""Import public GitHub data files without executing repository code."""
from pathlib import Path
import urllib.parse
import urllib.request
import uuid

MAX_BYTES = 50 * 1024 * 1024


def github_file_url(value: str) -> str:
    url = urllib.parse.urlsplit(value.strip())
    parts = urllib.parse.unquote(url.path).split("/")[1:]
    if url.scheme != "https" or url.netloc not in {"github.com", "raw.githubusercontent.com"} or any(part in {"", ".", ".."} or "\\" in part for part in parts):
        raise ValueError("Use an HTTPS GitHub file link or raw.githubusercontent.com link.")
    if url.netloc == "github.com":
        if len(parts) < 5 or parts[2] not in {"blob", "raw"}:
            raise ValueError("Open a CSV, JSONL, or Parquet file on GitHub and paste its file link, not a repository or folder link.")
        parts.pop(2)
    if len(parts) < 4 or Path(parts[-1]).suffix.lower() not in {".csv", ".jsonl", ".parquet"}:
        raise ValueError("Choose a CSV, JSONL, or Parquet file on GitHub.")
    return "https://raw.githubusercontent.com/" + "/".join(urllib.parse.quote(part, safe="") for part in parts)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def import_github_dataset(value: str, directory: Path) -> dict:
    url = github_file_url(value)
    filename = urllib.parse.unquote(url.rsplit("/", 1)[-1])
    destination = directory / (str(uuid.uuid4()) + Path(filename).suffix.lower())
    request = urllib.request.Request(url, headers={"User-Agent": "ForgeTune", "Accept": "application/octet-stream"})
    total = 0
    try:
        with urllib.request.build_opener(NoRedirect()).open(request, timeout=30) as response, destination.open("xb") as output:
            while chunk := response.read(65536):
                total += len(chunk)
                if total > MAX_BYTES:
                    raise ValueError("GitHub imports are limited to 50 MB. Download and prepare a smaller dataset first.")
                output.write(chunk)
        if total == 0:
            raise ValueError("The GitHub file is empty.")
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return {"source": "upload", "name": str(destination), "displayName": filename, "extension": destination.suffix[1:], "sourceUrl": url}
