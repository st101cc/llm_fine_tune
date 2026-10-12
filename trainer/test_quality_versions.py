import json
import pytest
from quality_store import QualityStore


def audited(root):
    store = QualityStore(root)
    path = store.datasets / 'source.jsonl'
    path.write_text(json.dumps({'prompt': 'question', 'completion': '软件'}) + '\n', encoding='utf-8')
    audit = store.create_audit({'source': 'upload', 'name': str(path)})
    store.run_audit(audit['id'])
    audit = store.get('audit', audit['id'])
    finding = next(f for f in store.findings(audit['id'])['items'] if f['editable'])
    store.review(audit['id'], audit['fingerprint'], [{'id': finding['id'], 'decision': 'accepted'}])
    return store, audit, finding


def test_version_report_retains_decisions_after_further_review(tmp_path):
    store, audit, finding = audited(tmp_path)
    version = store.materialize(audit['id'], audit['fingerprint'])
    store.review(audit['id'], audit['fingerprint'], [{'id': finding['id'], 'decision': 'ignored'}])
    assert store.version_findings(version['id'])['items'][0]['decision'] == 'accepted'


def test_corrupt_partial_owned_by_other_worker_is_preserved(tmp_path, monkeypatch):
    store, audit, _ = audited(tmp_path)
    from pathlib import Path
    original = Path.open
    written = []
    def competing(path, mode='r', *args, **kwargs):
        if mode == 'x':
            with original(path, 'w', encoding='utf-8') as stream:
                stream.write('other worker')
            written.append(path)
        return original(path, mode, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', competing)
    with pytest.raises(FileExistsError):
        store.materialize(audit['id'], audit['fingerprint'])
    assert written[0].read_text() == 'other worker'
