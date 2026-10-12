"""Convert saved ForgeTune answers without inventing absent legacy provenance."""


def saved_answers(record):
    state = record.get('state', record)
    task = state.get('profile', {}).get('classification', {}).get('task', 'writing')
    task = {'structured': 'json'}.get(task, task)
    if task not in ('classification', 'json', 'mcq'):
        task = 'writing'
    cases, answers, seen = {}, [], set()
    for field, phase in (('baseline_results', 'base'), ('evaluation_results', 'tuned')):
        for result in state.get(field, []):
            model = result.get('modelId')
            if not isinstance(model, str) or not model:
                continue
            candidate = model + ':' + phase
            for sample in result.get('samples', []):
                if sample.get('inputTruncated') or sample.get('hitTokenLimit'):
                    raise ValueError('Saved answers contain truncation; use fresh untruncated evaluation outputs.')
                row_id = sample.get('rowId')
                if type(row_id) is not int or row_id < 0 or not isinstance(sample.get('input'), str):
                    raise ValueError('Saved outputs need stable row IDs and input text.')
                case_id = str(row_id)
                case = {'id': case_id, 'input': sample['input'], 'reference': sample.get('reference'), 'task': task}
                if case_id in cases and cases[case_id] != case:
                    raise ValueError('Saved candidates disagree on evaluation cases; evaluate matching phases separately.')
                cases[case_id] = case
                output = sample.get('tunedOutput' if phase == 'tuned' else 'baseOutput')
                if not isinstance(output, str):
                    continue  # Keep the case visible as missing coverage.
                if (candidate, case_id) in seen:
                    raise ValueError('Duplicate saved candidate/case ID; choose a single experiment.')
                seen.add((candidate, case_id))
                answer = {'candidate': candidate, 'caseId': case_id, 'output': output, 'provenance': {'status': 'legacy', 'source': 'ForgeTune saved outputs'}}
                original = next((a for a in result.get('qualityGeneration', {}).get('answers', []) if a.get('caseId') == case_id and a.get('output') == output), None)
                if original and original.get('protocol') and original.get('provenance'):
                    answer.update(protocol=original['protocol'], provenance=original['provenance'])
                answers.append(answer)
    if not cases:
        raise ValueError('No saved answer cases available. Upload case and answer JSONL files instead.')
    return list(cases.values()), answers
