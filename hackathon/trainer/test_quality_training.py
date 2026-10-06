import importlib.util
import pytest
from test_quality_versions import audited


def test_quality_handoff_validates_derivative_and_preserves_lineage(tmp_path):
    assert importlib.util.find_spec('quality_training'), 'Quality training handoff missing'
    from quality_training import dataset_provenance
    store, audit, _ = audited(tmp_path)
    version = store.materialize(audit['id'], audit['fingerprint'])
    assert dataset_provenance(tmp_path, version['dataset']) == version['provenance']
    assert dataset_provenance(tmp_path, {'source': 'upload', 'name': 'legacy'}) is None
    ref = {**version['dataset'], 'name': audit['dataset']['name']}
    with pytest.raises(ValueError, match='match'):
        dataset_provenance(tmp_path, ref)
    store.export_path(version['id']).write_text('changed', encoding='utf-8')
    with pytest.raises(ValueError, match='changed'):
        dataset_provenance(tmp_path, version['dataset'])


def test_pilot_candidate_never_substitutes_or_downloads(monkeypatch):
    import quality_models
    import quality_training
    monkeypatch.setattr(quality_models, 'get_model_status', lambda: {'status': 'missing_weights', 'message': 'Prepare TAIDE first'})
    with pytest.raises(ValueError, match='Prepare TAIDE'):
        quality_training.pilot_candidate(24)
    monkeypatch.setattr(quality_models, 'get_model_status', lambda: {'status': 'ready', 'path': '/cache/sha', 'revision': 'a' * 40, 'license': {'id': 'license'}})
    with pytest.raises(ValueError, match='memory'):
        quality_training.pilot_candidate(4)
    candidate = quality_training.pilot_candidate(24)
    assert candidate['model']['id'] == quality_models.TAIDE_ID
    assert candidate['model']['localPath'] == '/cache/sha'


def test_workflow_inference_uses_bounded_local_runner(tmp_path, monkeypatch):
    from datasets import Dataset
    import quality_pilot
    import quality_training
    called = []
    def generate(cases, **kwargs):
        called.append(kwargs)
        return {'status': 'complete', 'answers': [{'caseId': '0', 'candidate': 'TAIDE', 'output': 'yes', 'protocol': {'v': 1}}]}
    monkeypatch.setattr(quality_pilot, 'generate_bounded', generate)
    result = quality_training.bounded_comparison(tmp_path, Dataset.from_list([{'text': 'q', 'label': 'yes'}]), [0], 'classification')
    assert result['qualityEvidence']['score'] == 1
    assert called[0]['runtime_root'] == tmp_path
    assert result['evaluationIds'] == [0]
