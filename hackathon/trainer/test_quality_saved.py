import importlib.util
import pytest


def test_legacy_outputs_have_stable_ids_without_fabricated_protocols():
    assert importlib.util.find_spec('quality_saved'), 'Saved output conversion missing'
    from quality_saved import saved_answers
    sample = {'rowId': 5, 'input': '分類?', 'reference': '正確', 'baseOutput': '正確'}
    record = {'state': {'profile': {'classification': {'task': 'classification'}}, 'baseline_results': [{'modelId': 'base', 'evaluationIds': [5], 'samples': [sample]}]}}
    cases, answers = saved_answers(record)
    assert cases == [{'id': '5', 'input': '分類?', 'reference': '正確', 'task': 'classification'}]
    assert answers[0]['caseId'] == '5'
    assert answers[0]['provenance']['status'] == 'legacy'
    assert 'protocol' not in answers[0]
    record['state']['baseline_results'][0]['samples'] *= 2
    with pytest.raises(ValueError, match='Duplicate'):
        saved_answers(record)


def test_saved_output_case_conflicts_and_truncation_rejected():
    assert importlib.util.find_spec('quality_saved')
    from quality_saved import saved_answers
    record = {'state': {'baseline_results': [{'modelId': 'a', 'samples': [{'rowId': 0, 'input': 'a', 'baseOutput': 'x', 'inputTruncated': True}]}]}}
    with pytest.raises(ValueError, match='truncat'):
        saved_answers(record)
