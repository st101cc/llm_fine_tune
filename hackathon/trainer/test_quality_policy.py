import importlib.util


def policy_module():
    assert importlib.util.find_spec('quality_policy'), 'Traditional Chinese policy is not implemented'
    import quality_policy
    return quality_policy


def test_suggests_script_and_regional_changes_without_mutating_text():
    q = policy_module()
    text = '请使用软件，並連接伺服器。'
    findings = q.inspect_text(text, {})
    assert any(f['replacement'] == '請' for f in findings)
    assert any('軟體' in (f['replacement'] or '') for f in findings)
    assert text == '请使用软件，並連接伺服器。'
    assert all(text[f['start']:f['end']] == f['original'] for f in findings if f['editable'])


def test_protected_text_and_shared_characters_are_not_rewritten():
    q = policy_module()
    for text in ['一人一手', '干面', '`软件`', '```\n软件\n```', '「软件」', '"软件"', 'https://example.com/软件']:
        assert not [f for f in q.inspect_text(text, {}) if f['editable']], text
    assert not q.inspect_text('软件', {'exclusions': ['软件']})


def test_customer_terms_are_preferences_and_ignore_entries_win():
    q = policy_module()
    findings = q.inspect_text('鼠標', {'terms': {'鼠標': '滑鼠'}})
    assert findings[0]['code'] == 'terminology'
    assert findings[0]['replacement'] == '滑鼠'
    assert not q.inspect_text('鼠標', {'terms': {'鼠標': '滑鼠'}, 'exclusions': ['鼠標']})


def test_only_target_fields_are_editable_and_raw_text_requires_selection():
    q = policy_module()
    fields = q.text_fields({'prompt': '软件', 'completion': '软件', 'label': '软件'}, {})
    assert [(p, editable) for p, text, editable in fields] == [(['prompt'], False), (['completion'], True)]
    assert q.text_fields({'text': '软件'}, {}) == [(['text'], '软件', False)]
    assert q.text_fields({'text': '软件'}, {'fields': ['text']}) == [(['text'], '软件', True)]
    fields = q.text_fields({'messages': [{'role': 'user', 'content': '软件'}, {'role': 'assistant', 'content': '软件'}]}, {})
    assert fields[-1] == (['messages', 1, 'content'], '软件', True)


def test_invalid_policy_and_malformed_training_rows_fail_explicitly():
    q = policy_module()
    import pytest
    with pytest.raises(ValueError):
        q.normalize_policy({'terms': {'': 'bad'}})
    assert q.row_problem({'messages': [{'role': 'assistant', 'content': 42}]})
    assert q.row_problem({'prompt': 'question', 'completion': ''})
    assert q.row_problem({'text': 'valid'}) is None
