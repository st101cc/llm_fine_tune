import io
import pytest
from github_import import github_file_url, import_github_dataset


def test_github_links_are_normalized():
    assert github_file_url("https://github.com/acme/data/blob/main/train.csv?raw=true") == "https://raw.githubusercontent.com/acme/data/main/train.csv"
    assert github_file_url("https://raw.githubusercontent.com/acme/data/main/train.jsonl").endswith("train.jsonl")
    for url in ["http://github.com/a/b/blob/main/a.csv", "https://evil.com/a.csv", "https://github.com/a/b", "https://github.com/a/b/blob/main/test.py", "https://github.com@evil.com/a.csv", "https://raw.githubusercontent.com/a/b/main/../a.csv"]:
        with pytest.raises(ValueError): github_file_url(url)


def test_import_is_bounded_and_preserves_source(tmp_path, monkeypatch):
    import github_import as module
    class Opener:
        def open(self, *args, **kwargs): return io.BytesIO(b'prompt,completion\nhi,hello\n')
    monkeypatch.setattr(module.urllib.request, "build_opener", lambda *args: Opener())
    result = import_github_dataset("https://github.com/a/b/blob/main/train.csv", tmp_path)
    assert result["source"] == "upload"
    assert result["sourceUrl"].startswith("https://raw.githubusercontent.com/")
    assert module.Path(result["name"]).read_bytes().startswith(b"prompt,completion")
    monkeypatch.setattr(module, "MAX_BYTES", 5)
    with pytest.raises(ValueError): import_github_dataset(result["sourceUrl"], tmp_path)
    assert len(list(tmp_path.iterdir())) == 1
