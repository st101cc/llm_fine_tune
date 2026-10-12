import importlib.util
import json
import time

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
