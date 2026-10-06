import importlib.util
import json
import pytest


def setup(tmp_path, rows):
    assert importlib.util.find_spec('quality_store'), 'Quality persistence is not implemented'
    from quality_store import QualityStore
    directory = tmp_path / 'datasets'
    directory.mkdir(exist_ok=True)
    path = directory / 'fixture.jsonl'
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    store = QualityStore(tmp_path)
    ref = {'source': 'upload', 'name': str(path), 'extension': 'jsonl'}
    return store, path, ref


def test_full_scan_covers_late_rows_and_sample_never_claims_full(tmp_path):
    rows = [{'prompt': str(i), 'completion': '正常文字'} for i in range(700)]
    rows[650]['completion'] = '软件'
    store, path, ref = setup(tmp_path, rows)
    audit = store.create_audit(ref, {}, 'full')
    store.run_audit(audit['id'])
    result = store.get('audit', audit['id'])
    assert result['status'] == 'complete' and result['scannedRows'] == result['totalRows'] == 700
    assert any(f['rowId'] == 650 for f in store.findings(audit['id'], limit=100)['items'])
    sampled = store.create_audit(ref, {}, 'sample')
    store.run_audit(sampled['id'])
    assert store.get('audit', sampled['id'])['coverage'] == 'sample'


def test_review_derivative_preserves_source_metadata_and_is_idempotent(tmp_path):
    row = {'prompt': '软件', 'completion': '软件', 'owner': 'original'}
    store, path, ref = setup(tmp_path, [row])
    original = path.read_bytes()
    audit = store.create_audit(ref, {}, 'full')
    store.run_audit(audit['id'])
    audit = store.get('audit', audit['id'])
    findings = store.findings(audit['id'])['items']
    editable = next(f for f in findings if f['editable'])
    store.review(audit['id'], audit['fingerprint'], [{'id': editable['id'], 'decision': 'accepted'}])
    version = store.materialize(audit['id'], audit['fingerprint'])
    output = json.loads(store.export_path(version['id']).read_text(encoding='utf-8'))
    assert output == {**row, 'completion': '軟體'}
    assert path.read_bytes() == original
    assert store.materialize(audit['id'], audit['fingerprint'])['id'] == version['id']
    assert version['provenance']['sourceFingerprint'] == audit['fingerprint']
    with pytest.raises(ValueError):
        store.review(audit['id'], audit['fingerprint'], [{'id': next(f['id'] for f in findings if not f['editable']), 'decision': 'accepted'}])


def test_stale_source_and_path_escape_are_rejected(tmp_path):
    store, path, ref = setup(tmp_path, [{'text': '软件'}])
    audit = store.create_audit(ref, {'fields': ['text']}, 'full')
    store.run_audit(audit['id'])
    fingerprint = store.get('audit', audit['id'])['fingerprint']
    path.write_text('{"text":"changed"}\n', encoding='utf-8')
    with pytest.raises(ValueError, match='changed|stale'):
        store.materialize(audit['id'], fingerprint)
    with pytest.raises(ValueError):
        store.create_audit({**ref, 'name': str(tmp_path / '..' / 'secret.jsonl')}, {}, 'full')
    with pytest.raises(ValueError):
        store.export_path('../secret')


def test_duplicates_conflicts_malformed_and_small_data_are_auditable(tmp_path):
    rows = [{'prompt':'a','completion':'one'}, {'prompt':'a','completion':'one'}, {'prompt':'a','completion':'two'}, {'prompt':'x','completion':''}]
    store, path, ref = setup(tmp_path, rows)
    audit = store.create_audit(ref, {}, 'full')
    store.run_audit(audit['id'])
    codes = {f['code'] for f in store.findings(audit['id'])['items']}
    assert {'duplicate', 'conflicting_answer', 'malformed'} <= codes
    assert store.get('audit', audit['id'])['status'] == 'complete'


def test_cancel_and_restart_never_report_pass_and_pagination_is_stable(tmp_path):
    store, path, ref = setup(tmp_path, [{'text': '软件'}] * 20)
    audit = store.create_audit(ref, {}, 'full')
    store.run_audit(audit['id'], cancelled=lambda: True)
    assert store.get('audit', audit['id'])['status'] == 'cancelled'
    pending = store.create_audit(ref, {}, 'full')
    store.recover()
    assert store.get('audit', pending['id'])['status'] == 'interrupted'
    audit = store.create_audit(ref, {}, 'full')
    store.run_audit(audit['id'])
    first, second = store.findings(audit['id'], limit=2), store.findings(audit['id'], offset=2, limit=2)
    assert len(first['items']) == len(second['items']) == 2
    assert not ({f['id'] for f in first['items']} & {f['id'] for f in second['items']})
