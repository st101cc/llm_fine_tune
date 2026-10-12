import asyncio
import json
import uuid
import pytest
from test_quality_versions import audited


def test_managed_derivative_cannot_drop_quality_identity(tmp_path):
    from quality_training import dataset_provenance
    store, audit, _ = audited(tmp_path)
    version = store.materialize(audit['id'], audit['fingerprint'])
    ref = dict(version['dataset'])
    ref.pop('qualityVersionId')
    with pytest.raises(ValueError, match='provenance'):
        dataset_provenance(tmp_path, ref)


def test_quality_workflow_cannot_launch_legacy_judge(monkeypatch):
    import app
    from fastapi import HTTPException
    record = {'id': str(uuid.uuid4()), 'status': 'complete', 'state': {'quality_provenance': {'auditId': 'a'}, 'baseline_results': [{}], 'evaluation_results': [{}]}}
    monkeypatch.setattr(app, 'read_workflow', lambda _: record)
    monkeypatch.setattr(app, 'active_evaluations', {})
    monkeypatch.setattr(app, 'active_workflows', {})
    monkeypatch.setattr(app, 'queue_lock', asyncio.Lock())
    monkeypatch.setattr(app, 'workflow_path', lambda _: (_ for _ in ()).throw(AssertionError('No legacy regrade writes allowed')))
    with pytest.raises(HTTPException) as exc:
        asyncio.run(app.evaluate_workflow(uuid.UUID(record['id'])))
    assert exc.value.status_code == 422


def test_missing_source_policy_prevents_pairing():
    from quality_evaluation import evaluate
    result = evaluate([{'id': 'q', 'input': '?', 'task': 'classification', 'reference': 'yes'}], [
        {'caseId': 'q', 'candidate': 'a', 'output': 'no', 'protocol': {'v': 1}},
        {'caseId': 'q', 'candidate': 'b', 'output': 'yes', 'protocol': {'v': 1}},
    ], {})
    assert result['comparisons'] == []
    assert result['unpaired'][0]['reason'] == 'policy provenance missing'


def test_cancelled_quality_training_never_reenters_budget(tmp_path, monkeypatch):
    import app
    import quality_pilot
    record = {'id': 'test', 'status': 'cancelled', 'qualityProvenance': {'auditId': 'a'}}
    monkeypatch.setattr(app, 'read_run', lambda _: record.copy())
    monkeypatch.setattr(app, 'queue_lock', asyncio.Lock())
    monkeypatch.setattr(app, 'write_run', lambda _: (_ for _ in ()).throw(AssertionError('Cancelled record must remain untouched')))
    asyncio.run(app.run_training('test'))


def test_protocol_mismatch_cannot_recommend_quality_tuning(tmp_path):
    from graph import _compare_results
    from quality_evaluation import evaluate, fingerprint
    from quality_policy import normalize_policy
    policy_hash = fingerprint(normalize_policy({}))
    cases = [{'id': '0', 'input': 'q', 'task': 'classification', 'reference': 'yes'}]
    def result(output, protocol):
        answers = [{'candidate': 'model', 'caseId': '0', 'output': output, 'protocol': protocol, 'policyHash': policy_hash}]
        report = evaluate(cases, answers, {})
        return {'modelId': 'model', 'status': 'complete', 'artifactValid': True, 'evaluationIds': [0], 'qualityReport': report, 'qualityEvidence': {'metric': 'classification_exact_match', 'score': float(output == 'yes'), 'coverage': 1, 'evaluatedExamples': 1, 'evaluatorId': 'same'}}
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps({'evaluationIds': [0]}))
    comparison = _compare_results({'quality_provenance': {'auditId': 'a'}, 'split_manifest_path': str(path), 'approved_plan': {'candidates': [{'model': {'id': 'model'}}]}, 'baseline_results': [result('no', {'template': 1})], 'evaluation_results': [result('yes', {'template': 2})]})
    assert comparison['pairs'][0]['qualityScoreDelta'] is None
    assert comparison['decision'] == 'review_required'
    from graph import diagnose_trials
    diagnosis = diagnose_trials([{'developmentIds': [0], 'results': [result('no', {'template': 1})]}, {'developmentIds': [0], 'results': [result('yes', {'template': 2})]}])
    assert diagnosis['selectedTrial'] == 0


def test_controlled_pilot_outputs_attest_the_selected_policy():
    import quality_pilot
    cases = [{'id': 'q', 'input': '?', 'task': 'classification', 'reference': 'yes'}]
    answers = [{'caseId': 'q', 'candidate': c, 'output': 'yes', 'protocol': {'v': 1}} for c in ('base', 'original', 'cleaned')]
    report = quality_pilot.evaluate_controlled_outputs(cases, answers, {})
    assert len(report['comparisons']) == 3
    assert all('policyHash' not in a for a in answers)
