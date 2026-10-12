import asyncio
import app
import quality_pilot


def test_quality_training_is_budgeted_without_launching_legacy_process(tmp_path, monkeypatch):
    record = {'id': 'test', 'status': 'queued', 'qualityProvenance': {'auditId': 'audit'}, 'config': {}}
    monkeypatch.setattr(app, 'read_run', lambda _: dict(record))
    monkeypatch.setattr(app, 'write_run', lambda value: record.update(value))
    monkeypatch.setattr(app, 'run_path', lambda _: tmp_path / 'run.json')
    monkeypatch.setattr(app, 'DATA_ROOT', tmp_path)
    monkeypatch.setattr(app, 'queue_lock', asyncio.Lock())
    calls = []
    def bounded(root, commands, **kwargs):
        calls.append((root, commands))
        return {'status': 'unavailable', 'message': 'GPU unavailable', 'chargedGpuSeconds': 0}
    monkeypatch.setattr(quality_pilot, 'run_with_pilot_budget', bounded)
    async def forbidden(*args, **kwargs):
        raise AssertionError('Legacy unbounded training process must not launch')
    monkeypatch.setattr(asyncio, 'create_subprocess_exec', forbidden)
    asyncio.run(app.run_training('test'))
    assert calls
    assert record['status'] == 'failed'
    assert record['qualityBudget']['chargedGpuSeconds'] == 0
