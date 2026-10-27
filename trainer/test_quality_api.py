import importlib.util
import json
import time
import uuid

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def setup_client(root):
    assert importlib.util.find_spec('quality_api'), 'Quality API is not implemented'
    from quality_api import install_quality_routes
    app = FastAPI()
    install_quality_routes(app, lambda: root)
    directory = root / 'datasets'
    directory.mkdir(exist_ok=True)
    path = directory / 'example.jsonl'
    path.write_text(json.dumps({'prompt': 'question', 'completion': '软件'}) + '\n', encoding='utf-8')
    return TestClient(app), {'source': 'upload', 'name': str(path), 'extension': 'jsonl'}


def finished(client, path):
    for _ in range(150):
        result = client.get(path).json()
        if result['status'] not in ('queued', 'running'):
            return result
        time.sleep(.03)
    raise AssertionError('Job did not finish')


def test_audit_review_export_without_gpu_or_workflow(tmp_path):
    client, dataset = setup_client(tmp_path)
    with client:
        response = client.post('/quality/audits', json={'dataset': dataset})
        assert response.status_code == 202
        url = '/quality/audits/' + response.json()['id']
        job = finished(client, url)
        assert job['status'] == 'complete'
        findings = client.get(url + '/findings').json()['items']
        finding = next(item for item in findings if item['editable'])
        response = client.post(url + '/review', json={'fingerprint': job['fingerprint'], 'decisions': [{'id': finding['id'], 'decision': 'accepted'}]})
        assert response.status_code == 200
        response = client.post(url + '/versions', json={'fingerprint': job['fingerprint']})
        assert response.status_code == 200, response.text
        export = '/quality/versions/' + response.json()['id'] + '/export'
        assert json.loads(client.get(export).text)['completion'] == '軟體'
        assert 'sourceFingerprint' in client.get(export + '?kind=report').text
        assert client.get('/quality/versions/not-an-id/export').status_code == 422


def test_saved_answer_evaluation_without_training(tmp_path):
    client, _ = setup_client(tmp_path)
    with client:
        response = client.post('/quality/evaluations', json={
            'cases': [{'id': 'a', 'input': 'a?', 'task': 'classification', 'reference': 'yes'}, {'id': 'b', 'input': 'b?', 'task': 'classification', 'reference': 'no'}],
            'answers': [{'caseId': 'a', 'candidate': 'base', 'output': 'yes'}]})
        assert response.status_code == 202, response.text
        url = '/quality/evaluations/' + response.json()['id']
        result = finished(client, url)
        assert result['status'] == 'complete', result
        assert result['result']
        assert client.get(url + '/export').status_code == 200


def test_external_paths_and_invalid_ids_rejected(tmp_path):
    client, _ = setup_client(tmp_path)
    with client:
        assert client.post('/quality/audits', json={'dataset': {'source': 'upload', 'name': 'C:/Windows/win.ini'}}).status_code == 422
        assert client.post('/quality/audits', json={'dataset': {'source': 'huggingface', 'name': 'remote/data'}}).status_code == 422
        assert client.get('/quality/audits/not-an-id').status_code == 422


def test_local_generation_unavailable_is_not_success(tmp_path, monkeypatch):
    import quality_pilot
    monkeypatch.setattr(quality_pilot, 'generate_bounded', lambda *a, **k: {'status': 'unavailable', 'message': 'TAIDE weights missing', 'answers': [], 'budget': {'chargedGpuSeconds': 0}})
    client, _ = setup_client(tmp_path)
    with client:
        response = client.post('/quality/evaluations', json={'cases': [{'id': 'a', 'input': 'q', 'task': 'writing'}], 'generate': True})
        assert response.status_code == 202, response.text
        record = finished(client, '/quality/evaluations/' + response.json()['id'])
        assert record['status'] == 'unavailable'
        assert 'result' not in record


def test_writing_sample_tokens_require_generation_and_reach_the_generator(tmp_path, monkeypatch):
    import quality_pilot
    seen = {}
    monkeypatch.setattr(quality_pilot, 'generate_bounded', lambda *a, **k: seen.update(k) or {'status': 'unavailable', 'message': 'stub', 'answers': [], 'budget': {'chargedGpuSeconds': 0}})
    client, _ = setup_client(tmp_path)
    case = {'id': 'w', 'input': 'q', 'task': 'writing'}
    with client:
        assert client.post('/quality/evaluations', json={'cases': [case], 'writingSampleTokens': 64}).status_code >= 400
        assert client.post('/quality/evaluations', json={'cases': [case], 'generate': True, 'writingSampleTokens': 999}).status_code == 422
        response = client.post('/quality/evaluations', json={'cases': [case], 'generate': True, 'writingSampleTokens': 64})
        assert response.status_code == 202, response.text
        record = finished(client, '/quality/evaluations/' + response.json()['id'])
    assert record['writingSampleTokens'] == 64 and seen['writing_sample_tokens'] == 64


def reviewed_version(root, rows):
    from quality_store import QualityStore
    directory = root / 'datasets'
    directory.mkdir(exist_ok=True)
    path = directory / 'pair.jsonl'
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    store = QualityStore(root)
    audit = store.create_audit({'source': 'upload', 'name': str(path), 'extension': 'jsonl', 'displayName': 'pair'}, {'terms': {'數據': '資料'}}, 'full')
    store.run_audit(audit['id'])
    audit = store.get('audit', audit['id'])
    store.accept_customer_terms(audit['id'], audit['fingerprint'])
    return store, path, store.materialize(audit['id'], audit['fingerprint'])


def character_tokens(monkeypatch):
    import comparison_job
    monkeypatch.setattr(comparison_job, 'taide_tokens', lambda row: len(json.dumps(row, ensure_ascii=False)))


TINY = [{'prompt': 'a', 'completion': '這是數據'}, {'prompt': 'b', 'completion': '好'}]
LARGE = [{'prompt': f'問題{i}', 'completion': '這是數據' if i % 2 else '好'} for i in range(300)]


def test_estimate_and_bounds(tmp_path, monkeypatch):
    character_tokens(monkeypatch)
    _, _, version = reviewed_version(tmp_path, TINY)
    client, _ = setup_client(tmp_path)
    with client:
        assert client.post('/quality/comparisons/estimate', json={'versionId': version['id'], 'epochs': 4}).status_code == 422
        assert client.post('/quality/comparisons/estimate', json={'versionId': version['id'], 'trainRows': 50}).status_code == 422
        response = client.post('/quality/comparisons/estimate', json={'versionId': version['id']})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body['feasible'] is False and body['split']['rows'] == 2
        assert any('training rows' in r for r in body['split']['reasons'])
        response = client.post('/quality/comparisons', json={'versionId': version['id']})
        assert response.status_code == 422 and 'training rows' in response.json()['detail']
        assert client.get('/quality/comparisons').json() == []


def test_create_conflict_and_cancel(tmp_path, monkeypatch):
    import threading
    import comparison_job
    character_tokens(monkeypatch)
    started = threading.Event()

    def blocking(store, root, comparison_id, stop, **_):
        started.set()
        stop.wait(10)
        record = store.get('comparison', comparison_id)
        record['status'] = 'cancelled' if record['status'] == 'cancelling' else 'running'
        store.put('comparison', record)
    monkeypatch.setattr(comparison_job, 'run_comparison', blocking)
    _, _, version = reviewed_version(tmp_path, LARGE)
    client, _ = setup_client(tmp_path)
    with client:
        estimate = client.post('/quality/comparisons/estimate', json={'versionId': version['id'], 'epochs': 1}).json()
        assert estimate['feasible'] is True, estimate
        response = client.post('/quality/comparisons', json={'versionId': version['id'], 'epochs': 1})
        assert response.status_code == 202, response.text
        record = response.json()
        assert record['status'] == 'queued' and record['settings']['epochs'] == 1
        assert started.wait(5)
        conflict = client.post('/quality/comparisons', json={'versionId': version['id']})
        assert conflict.status_code == 409 and conflict.json()['detail'] == 'Another comparison is running'
        listed = client.get('/quality/comparisons').json()
        assert [(c['id'], c['displayName'], c['headline']) for c in listed] == [(record['id'], version['dataset']['displayName'], None)]
        url = '/quality/comparisons/' + record['id']
        assert client.post(url + '/cancel').json()['status'] == 'cancelling'
        for _ in range(150):
            if client.get(url).json()['status'] == 'cancelled':
                break
            time.sleep(.03)
        assert client.get(url).json()['status'] == 'cancelled'
        export = client.get(url + '/export')
        assert export.status_code == 200 and f'comparison-{record["id"]}.json' in export.headers['content-disposition']
        assert export.json()['comparison']['id'] == record['id']


def test_startup_resumes_running_comparison(tmp_path, monkeypatch):
    import threading
    import comparison_job
    from quality_store import QualityStore
    calls, stopped = [], []

    def recording(store, root, comparison_id, stop, **_):
        calls.append(comparison_id)
        stopped.append(stop.wait(10))
    monkeypatch.setattr(comparison_job, 'run_comparison', recording)
    store = QualityStore(tmp_path)
    running = store.put('comparison', comparison_job.new_record('v', {}) | {'status': 'running'})
    cancelling = store.put('comparison', comparison_job.new_record('v', {}) | {'status': 'cancelling'})
    store.put('comparison', comparison_job.new_record('v', {}) | {'status': 'complete'})
    client, _ = setup_client(tmp_path)
    with client:
        for _ in range(100):
            if len(calls) == 2:
                break
            time.sleep(.03)
    assert sorted(calls) == sorted([running['id'], cancelling['id']])
    assert stopped == [True, True]
    assert store.get('comparison', running['id'])['status'] == 'running'


def test_missing_source_is_422_not_503(tmp_path, monkeypatch):
    character_tokens(monkeypatch)
    _, path, version = reviewed_version(tmp_path, TINY)
    path.unlink()
    client, _ = setup_client(tmp_path)
    with client:
        for route in ('/quality/comparisons/estimate', '/quality/comparisons'):
            response = client.post(route, json={'versionId': version['id']})
            assert response.status_code == 422, response.text
            assert response.json()['detail'] == 'Source dataset is missing; start a new audit.'


READY = {n: {'modelId': n, 'revision': n[0] * 40, 'status': 'ready', 'path': '/m'} for n in ('breeze', 'qwen', 'yi')}


@pytest.fixture
def judges(monkeypatch):
    import quality_models
    import panel_job
    import comparison_job
    monkeypatch.setattr(quality_models, 'judge_status', lambda root: READY)
    monkeypatch.setattr(panel_job, 'run_panel', lambda *args, **kwargs: None)
    monkeypatch.setattr(comparison_job, 'run_comparison', lambda *args, **kwargs: None)


def panel_client(tmp_path, cases=25, **kwargs):
    from quality_store import QualityStore
    from test_judge_panel import make_comparison
    store = QualityStore(tmp_path)
    comparison_id = make_comparison(store, cases=cases, **kwargs)
    return store, comparison_id, setup_client(tmp_path)[0]


def test_panel_estimate_for_comparison_defaults_to_20_prompts(tmp_path, judges):
    _, comparison_id, client = panel_client(tmp_path)
    with client:
        response = client.post('/quality/panels/estimate', json={'source': {'comparisonId': comparison_id}})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body['items'] == 60 and body['prompts'] == 20 and body['maxPrompts'] == 25 and body['referenceMode'] == 'none'
        assert body['candidates'] == ['base', 'original', 'reviewed'] and body['truncatedUnknown'] is False
        assert body['feasible'] is True and body['reasons'] == [] and body['estimate']['gradesPerJudge'] == 60
        assert body['judges'] == READY and body['selfGrading'] == {}
        body = client.post('/quality/panels/estimate', json={'source': {'comparisonId': comparison_id}, 'prompts': 100}).json()
        assert body['prompts'] == 25 and body['items'] == 75
        assert client.post('/quality/panels/estimate', json={'source': {'comparisonId': comparison_id}, 'prompts': 0}).status_code == 422


def test_panel_estimate_refuses_missing_judge(tmp_path, judges, monkeypatch):
    import quality_models
    monkeypatch.setattr(quality_models, 'judge_status', lambda root: {**READY, 'yi': {**READY['yi'], 'status': 'missing_weights'}})
    _, comparison_id, client = panel_client(tmp_path)
    with client:
        body = client.post('/quality/panels/estimate', json={'source': {'comparisonId': comparison_id}}).json()
        assert body['feasible'] is False and body['reasons'] == ['yi: missing_weights']
        response = client.post('/quality/panels', json={'source': {'comparisonId': comparison_id}})
        assert response.status_code == 422 and 'yi: missing_weights' in response.json()['detail']
        assert client.get('/quality/panels').json() == []


def test_panel_estimate_rejects_unknown_or_double_source(tmp_path, judges):
    _, comparison_id, client = panel_client(tmp_path)
    with client:
        for source in ({}, {'other': 'x'}, {'evaluationId': 'a', 'comparisonId': comparison_id}, {'comparisonId': comparison_id, 'extra': 1},
                       {'comparisonId': 'missing'}, {'evaluationId': 'missing'}, {'comparisonId': 5}):
            for route in ('/quality/panels/estimate', '/quality/panels'):
                assert client.post(route, json={'source': source}).status_code == 422, (route, source)


def test_panel_estimate_for_incomplete_comparison_is_422(tmp_path, judges):
    _, comparison_id, client = panel_client(tmp_path, status='running')
    with client:
        response = client.post('/quality/panels/estimate', json={'source': {'comparisonId': comparison_id}})
        assert response.status_code == 422 and response.json()['detail'] == 'comparison has no saved evaluation answers'


def test_panel_estimate_for_evaluation_flags_uploaded_answers(tmp_path, judges):
    from quality_store import QualityStore
    from test_judge_panel import make_evaluation
    evaluation_id = make_evaluation(QualityStore(tmp_path))
    with setup_client(tmp_path)[0] as client:
        body = client.post('/quality/panels/estimate', json={'source': {'evaluationId': evaluation_id}}).json()
        assert body['referenceMode'] == 'case' and body['truncatedUnknown'] is True and body['maxPrompts'] == 2 and body['items'] == 4
        assert body['selfGrading'] == {'Qwen2.5-7B@base': 'qwen'}


def test_panel_estimate_refuses_while_comparison_runs(tmp_path, judges):
    import comparison_job
    store, comparison_id, client = panel_client(tmp_path)
    store.put('comparison', comparison_job.new_record('v', {}) | {'status': 'running'})
    with client:
        body = client.post('/quality/panels/estimate', json={'source': {'comparisonId': comparison_id}}).json()
        assert body['feasible'] is False and body['estimate']['feasible'] is False
        assert body['reasons'] == ['a comparison is running; wait for it to finish']


def test_panel_estimate_refuses_while_panel_runs(tmp_path, judges):
    import panel_job
    store, comparison_id, client = panel_client(tmp_path)
    store.put('panel', panel_job.new_record({'comparisonId': comparison_id}, 20, 'other') | {'status': 'running'})
    with client:
        body = client.post('/quality/panels/estimate', json={'source': {'comparisonId': comparison_id}}).json()
        assert body['feasible'] is False and body['reasons'] == ['a judge panel is running; wait for it to finish']


def test_panel_and_comparison_are_mutually_exclusive(tmp_path, judges):
    import comparison_job
    from quality_store import QualityStore
    _, comparison_id, client = panel_client(tmp_path / 'a')
    source = {'source': {'comparisonId': comparison_id}}
    with client:
        response = client.post('/quality/panels', json=source)
        assert response.status_code == 202, response.text
        record = response.json()
        assert record['status'] == 'queued' and record['prompts'] == 20 and record['source'] == {'comparisonId': comparison_id}
        assert client.post('/quality/comparisons', json={'versionId': 'v'}).status_code == 409
        assert client.post('/quality/panels', json=source).status_code == 409
        assert [p['id'] for p in client.get('/quality/panels').json()] == [record['id']]
    (tmp_path / 'b').mkdir()
    store = QualityStore(tmp_path / 'b')
    store.put('comparison', comparison_job.new_record('v', {}) | {'status': 'running'})
    _, comparison_id, client = panel_client(tmp_path / 'b')
    with client:
        assert client.post('/quality/panels', json={'source': {'comparisonId': comparison_id}}).status_code == 409


def test_panel_export_shape_and_label(tmp_path, judges):
    store, comparison_id, client = panel_client(tmp_path)
    vote = {'correctness': {'verdict': 'pass', 'reason': 'ok'}, 'language': {'verdict': 'pass', 'reason': 'ok'}}
    with client:
        record = client.post('/quality/panels', json={'source': {'comparisonId': comparison_id}, 'prompts': 2}).json()
        folder = store.directory / 'panels' / record['id']
        folder.mkdir(parents=True)
        (folder / 'items.jsonl').write_text(json.dumps({'itemId': 'row:0|base'}) + '\n', encoding='utf-8')
        (folder / 'grades-breeze.jsonl').write_text(json.dumps({'itemId': 'row:0|base', 'vote': vote}) + '\n{"itemId": "row:', encoding='utf-8')
        protocol = {'panelVersion': 'judge-panel-v1', 'languageRubric': 'Pass natural zh-TW.'}
        store.put('panel', {**record, 'status': 'complete', 'report': {'incomplete': False}, 'protocol': protocol})
        listed = client.get('/quality/panels').json()
        assert listed == [{'id': record['id'], 'displayName': record['displayName'], 'status': 'complete', 'createdAt': record['createdAt'], 'source': record['source']}]
        assert client.get('/quality/panels/' + record['id']).json()['report'] == {'incomplete': False}
        export = client.get(f'/quality/panels/{record["id"]}/export')
        assert export.status_code == 200 and f'panel-{record["id"]}.json' in export.headers['content-disposition']
        body = export.json()
        assert set(body) == {'panel', 'items', 'grades', 'report', 'label'} and 'report' not in body['panel'] and body['panel']['id'] == record['id']
        assert body['items'] == [{'itemId': 'row:0|base'}] and body['report'] == {'incomplete': False}
        assert body['grades'] == {'breeze': [{'itemId': 'row:0|base', 'vote': vote}], 'qwen': [], 'yi': []}
        assert body['panel']['protocol'] == protocol and body['panel']['protocol']['panelVersion'] == 'judge-panel-v1' and body['panel']['protocol']['languageRubric']
        assert body['label'] == 'estimated by local models, not verified'
        assert client.get('/quality/panels/' + str(uuid.uuid4()) + '/export').status_code == 404


def test_panel_never_changes_evaluation_judgments(tmp_path, judges, monkeypatch):
    import panel_job
    from quality_store import QualityStore
    from test_judge_panel import make_evaluation
    store = QualityStore(tmp_path)
    evaluation_id = make_evaluation(store)

    def finish(target, root, panel_id, stop, **_):
        record = target.get('panel', panel_id)
        target.put('panel', {**record, 'status': 'complete', 'report': {'incomplete': False}})
    monkeypatch.setattr(panel_job, 'run_panel', finish)
    with setup_client(tmp_path)[0] as client:
        before = client.get('/quality/evaluations/' + evaluation_id).content
        record = client.post('/quality/panels', json={'source': {'evaluationId': evaluation_id}}).json()
        assert finished(client, '/quality/panels/' + record['id'])['status'] == 'complete'
        assert client.get('/quality/evaluations/' + evaluation_id).content == before


def test_startup_relaunches_running_panel(tmp_path, judges, monkeypatch):
    import panel_job
    from quality_store import QualityStore
    calls, stopped = [], []

    def recording(store, root, panel_id, stop, **_):
        calls.append(panel_id)
        stopped.append(stop.wait(10))
    monkeypatch.setattr(panel_job, 'run_panel', recording)
    store = QualityStore(tmp_path)
    ids = [store.put('panel', panel_job.new_record({'comparisonId': 'c'}, None, 'x') | {'status': status})['id'] for status in ('queued', 'running', 'cancelling', 'complete', 'failed')]
    with setup_client(tmp_path)[0]:
        for _ in range(100):
            if len(calls) == 3:
                break
            time.sleep(.03)
    assert sorted(calls) == sorted(ids[:3]) and stopped == [True] * 3
    assert store.get('panel', ids[1])['status'] == 'running'
