"""Verified provenance boundary for handoff to existing approval workflows."""
from pathlib import Path
from quality_store import QualityStore, identifier


def dataset_provenance(root, dataset):
    version_id = dataset.get('qualityVersionId')
    if not version_id:
        path = Path(dataset.get('name', '')).resolve()
        if dataset.get('source') == 'upload' and path.parent == (Path(root).resolve() / 'datasets'):
            try:
                inferred = identifier(path.stem)
                QualityStore(root).get('version', inferred)
            except (ValueError, KeyError):
                pass
            else:
                raise ValueError('Managed quality dataset is missing its required provenance reference.')
        return None  # Legacy, never backfill invented provenance.
    store = QualityStore(root)
    version = store.get('version', version_id)
    path = store.export_path(version_id)
    if dataset.get('source') != 'upload' or Path(dataset.get('name', '')).resolve() != path or dataset.get('extension') != 'jsonl':
        raise ValueError('Dataset reference does not match the approved version.')
    return version['provenance']


def pilot_candidate(memory_gb):
    from quality_models import TAIDE_ID, get_model_status
    status = get_model_status()
    if status['status'] != 'ready':
        raise ValueError(status['message'])
    if memory_gb < 14:
        raise ValueError('Insufficient GPU memory: TAIDE QLoRA pilot requires at least 14 GB free; actual fit must still be verified.')
    return {'model': {'id': TAIDE_ID, 'name': 'TAIDE', 'sizeB': 8.5, 'localPath': status['path'], 'revision': status['revision'], 'licenseMetadata': status['license']}, 'method': 'qlora', 'estimatedVramGb': 14, 'rejected': None, 'reasons': ['Explicit local TAIDE pilot; no model substitution.']}


def no_automatic_judge(*args, **kwargs):
    return '{"verdict":"abstain","reason":"Free-form correctness needs human review; no local judge was requested."}'


def bounded_comparison(root, dataset, indices, task, instruction='', adapter_path=None, knowledge_documents=None, *, revision=None, policy=None):
    import json
    from advisor import evaluation_messages
    from quality_evaluation import evaluate, fingerprint
    from quality_policy import normalize_policy
    from quality_models import TAIDE_ID
    from quality_pilot import generate_bounded
    cases = []
    task = {'structured': 'json'}.get(task, task)
    if task not in ('classification', 'json', 'mcq'):
        task = 'writing'
    for index in indices:
        row = dataset[index]
        prompt_instruction = instruction
        if knowledge_documents:
            from quality_probe import retrieval_instruction
            prompt_instruction = retrieval_instruction(row, instruction, knowledge_documents)[0]
        messages, reference = evaluation_messages(row, prompt_instruction)
        text = messages[0]['content'] if len(messages) == 1 and messages[0]['role'] == 'user' else json.dumps(messages, ensure_ascii=False)
        cases.append({'id': str(index), 'input': text, 'reference': reference or None, 'task': task})
    policy = normalize_policy(policy)
    generated = generate_bounded(cases, runtime_root=root, adapter_path=adapter_path, revision=revision)
    if generated['status'] != 'complete':
        return {'modelId': TAIDE_ID, 'status': 'unavailable', 'message': generated['message'], 'qualityGeneration': generated}
    for answer in generated['answers']:
        answer['policyHash'] = fingerprint(policy)  # Attest this actual server-side scoring policy.
    report = evaluate(cases, generated['answers'], policy)
    candidate = next(iter(report['candidates'].values()))
    metrics = candidate['metrics'][task]
    scored = metrics['scored']
    outputs = {a['caseId']: a['output'] for a in generated['answers']}
    return {'modelId': TAIDE_ID, 'status': 'complete', 'evaluationIds': list(indices),
            'qualityGeneration': generated, 'qualityReport': report,
            'qualityEvidence': {'metric': task + '_exact_match' if task != 'writing' else None, 'score': metrics['accuracy'], 'coverage': scored / len(cases) if cases else 0, 'evaluatedExamples': scored, 'totalExamples': len(cases), 'estimated': False, 'status': 'measured' if scored == len(cases) else 'needs_review', 'evaluatorId': report['provenance']['evaluatorVersion']},
            'samples': [{'rowId': int(c['id']), 'input': c['input'], 'reference': c['reference'], 'referenceAvailable': c['reference'] is not None, 'tunedOutput' if adapter_path else 'baseOutput': outputs[c['id']]} for c in cases if c['id'] in outputs]}


def quality_pair_compatible(base, tuned):
    left, right = base.get('qualityReport', {}), tuned.get('qualityReport', {})
    a, b = left.get('provenance', {}), right.get('provenance', {})
    if any(not a.get(k) or a.get(k) != b.get(k) for k in ('casesHash', 'policyHash', 'evaluatorVersion')):
        return False
    if len(left.get('candidates', {})) != 1 or len(right.get('candidates', {})) != 1:
        return False
    x, y = next(iter(left['candidates'].values())), next(iter(right['candidates'].values()))
    p, q = x['provenance'], y['provenance']
    return bool(x['coverage']['caseIds'] and x['coverage'] == y['coverage'] and
                p['protocolByCase'] == q['protocolByCase'] and None not in p['protocolByCase'].values() and
                p['sourcePolicyHashes'] == q['sourcePolicyHashes'] == [a['policyHash']] and
                [r['caseId'] for r in x['results'] if r['correct'] is not None] == [r['caseId'] for r in y['results'] if r['correct'] is not None])
