import time
import pytest
from quality_policy import inspect_text, normalize_policy
from test_quality_versions import audited


def test_code_quotes_and_adjacent_ambiguous_characters_are_protected():
    policy = normalize_policy({})
    for text in ['~~~text\n软件\n~~~', '    软件', '‘软件’', '``软件 ` x``']:
        assert not any(f['editable'] for f in inspect_text(text, policy)), text
    for finding in inspect_text('干面软件', policy):
        if finding['editable']:
            assert finding['start'] >= 2


def test_repetitive_fields_are_bounded_and_cancellable():
    started = time.monotonic()
    inspect_text('a软' * 1000, normalize_policy({}))
    assert time.monotonic() - started < 3
    with pytest.raises(InterruptedError):
        inspect_text('a软' * 1000, normalize_policy({}), cancelled=lambda: True)


def test_publish_record_failure_is_retryable(tmp_path, monkeypatch):
    store, audit, _ = audited(tmp_path)
    original = store.put
    def fail(kind, *args, **kwargs):
        if kind == 'version':
            raise OSError('disk failure')
        return original(kind, *args, **kwargs)
    monkeypatch.setattr(store, 'put', fail)
    with pytest.raises(OSError):
        store.materialize(audit['id'], audit['fingerprint'])
    monkeypatch.setattr(store, 'put', original)
    store.recover()
    version = store.materialize(audit['id'], audit['fingerprint'])
    assert store.export_path(version['id']).is_file()


def test_policy_versions_cannot_be_silently_relabelled():
    with pytest.raises(ValueError, match='Unsupported'):
        normalize_policy({'version': 'other-policy'})
    with pytest.raises(ValueError, match='Unsupported'):
        normalize_policy({'openccVersion': '0.0.0'})


def test_restart_recovers_publication_after_durable_intent(tmp_path, monkeypatch):
    import quality_store
    store, audit, _ = audited(tmp_path)
    replace = quality_store.os.replace
    def fail(*args):
        raise OSError('rename interrupted')
    monkeypatch.setattr(quality_store.os, 'replace', fail)
    with pytest.raises(OSError):
        store.materialize(audit['id'], audit['fingerprint'])
    assert store.list('version')[0]['status'] == 'publishing'
    monkeypatch.setattr(quality_store.os, 'replace', replace)
    store.recover()
    recovered = store.list('version')[0]
    assert recovered['status'] == 'complete'
    assert store.export_path(recovered['id']).is_file()


def test_many_ambiguous_positions_do_not_create_quadratic_work():
    started = time.monotonic()
    assert not inspect_text('干' * 16384, {})
    assert time.monotonic() - started < 3


def test_restart_quarantines_unrecorded_partial_without_deleting_it(tmp_path):
    import uuid
    from quality_store import QualityStore
    store = QualityStore(tmp_path)
    partial = store.datasets / (str(uuid.uuid4()) + '.jsonl.partial')
    partial.write_text('interrupted derivative', encoding='utf-8')
    store.recover()
    assert not partial.exists()
    recovered = list((store.directory / 'recovery-orphans').glob('*.partial'))
    assert len(recovered) == 1
    assert recovered[0].read_text(encoding='utf-8') == 'interrupted derivative'
