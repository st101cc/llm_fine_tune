"""Local-only quality API; no training or hosted services required."""
from typing import Literal
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
import threading
import uuid
import json

from fastapi import HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from quality_policy import normalize_policy
from quality_store import QualityStore, encoded, identifier, digest


class StrictRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')


class AuditRequest(StrictRequest):
    dataset: dict
    policy: dict = Field(default_factory=dict)
    mode: str = 'full'


class ReviewRequest(StrictRequest):
    fingerprint: str = Field(min_length=64, max_length=64)
    decisions: list[dict] = Field(default_factory=list, max_length=500)
    exclusions: list[str] = Field(default_factory=list, max_length=1000)


class VersionRequest(StrictRequest):
    fingerprint: str = Field(min_length=64, max_length=64)


class EvaluationRequest(StrictRequest):
    loadMode: Literal["fp16", "bnb4"] = "fp16"
    cases: list[dict] = Field(default_factory=list, max_length=50000)
    answers: list[dict] = Field(default_factory=list, max_length=150000)
    policy: dict = Field(default_factory=dict)
    generate: bool = False
    benchmarkRevision: str | None = None
    allSubjects: bool = False
    adapterRunId: str | None = None
    maxGpuSeconds: int = Field(default=300, ge=1, le=1800)


def guarded(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except KeyError:
            raise HTTPException(404, 'Quality record not found.') from None
        except (ValueError, TypeError) as error:
            raise HTTPException(422, str(error)) from None
        except OSError:
            raise HTTPException(503, 'Local quality storage is unavailable. Check disk space and permissions.') from None
    return wrapped


def install_quality_routes(app, get_root):
    stores, jobs = {}, {}
    lock = threading.RLock()
    pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix='quality')

    def store():
        root = str(Path(get_root()).resolve())
        with lock:
            if root not in stores:
                stores[root] = QualityStore(root)
                stores[root].recover()
            return stores[root]

    def start(kind, record, worker):
        with lock:
            if len(jobs) >= 8:
                record.update(status='rejected', message='Quality queue was full; submit a new job.')
                store().put(kind, record)
                raise HTTPException(429, 'Quality queue is full; wait for a job or cancel one.')
            event = threading.Event()
            jobs[record['id']] = event

        def run():
            try:
                worker(event)
            finally:
                with lock:
                    jobs.pop(record['id'], None)
        pool.submit(run)
        return record

    @app.on_event('startup')
    def recover():
        store()

    @app.on_event('shutdown')
    def shutdown():
        with lock:
            for event in jobs.values():
                event.set()
        pool.shutdown(wait=True)
        for item in stores.values():
            item.recover()

    @app.get('/quality/audits')
    @guarded
    def list_audits():
        return store().list('audit')

    @app.get('/quality/sources')
    @guarded
    def sources():
        target = store()
        found = {}
        def add(dataset, label=None):
            if not isinstance(dataset, dict):
                return
            try:
                path = target.source_path(dataset)
            except (ValueError, OSError):
                return
            found[str(path)] = {'dataset': dataset, 'label': label or dataset.get('displayName') or path.name}
        for audit in target.list('audit'):
            add(audit['dataset'])
        for version in target.list('version'):
            add(version['dataset'])
        for path in (target.root / 'workflows').glob('*/workflow.json'):
            try:
                record = json.loads(path.read_text(encoding='utf-8'))
                add(record.get('state', {}).get('dataset'))
            except (ValueError, OSError):
                continue
        for path in (target.datasets / '.imports').glob('*.json'):
            try:
                result = json.loads(path.read_text(encoding='utf-8')).get('result') or {}
                if result.get('allSubsets'):
                    for subset in result.get('subsets', []):
                        if subset.get('status') == 'analyzed':
                            add(subset.get('dataset'), subset.get('configuration'))
                else:
                    add(result)
            except (ValueError, OSError):
                continue
        return list(found.values())

    @app.get('/quality/models')
    @guarded
    def model_status():
        from quality_models import get_model_status
        return get_model_status()

    @app.get('/quality/budget')
    @guarded
    def budget():
        from quality_pilot import budget_status
        return budget_status(store().root)

    @app.get('/quality/benchmarks')
    @guarded
    def benchmarks():
        from quality_models import load_benchmark
        result = []
        root = store().root
        for path in (root / 'quality' / 'benchmarks' / 'tmmluplus').glob('*/*.json'):
            if path.name not in ('default-30.json', 'all-subjects.json'):
                continue
            try:
                item = load_benchmark(path.parent.name, runtime_root=root, all_subjects=path.name == 'all-subjects.json')
                result.append({key: value for key, value in item.items() if key not in ('cases', 'files')})
            except ValueError:
                result.append({'revision': path.parent.name, 'status': 'invalid', 'message': 'Prepared benchmark failed integrity checks.'})
        return result

    @app.get('/quality/workflows/{workflow_id}/answers')
    @guarded
    def workflow_answers(workflow_id: str):
        from quality_saved import saved_answers
        root = store().root
        path = root / 'workflows' / identifier(workflow_id) / 'workflow.json'
        if not path.resolve().is_relative_to(root) or not path.is_file():
            raise KeyError('Workflow not found.')
        cases, answers = saved_answers(json.loads(path.read_text(encoding='utf-8')))
        return {'cases': cases, 'answers': answers, 'provenanceStatus': 'recorded' if all(a.get('protocol') for a in answers) else 'legacy-or-mixed', 'message': 'Absent scoring protocols are not inferred; incompatible comparisons remain unavailable.'}

    @app.post('/quality/audits', status_code=202)
    @guarded
    def create_audit(body: AuditRequest):
        target = store()
        record = target.create_audit(body.dataset, body.policy, body.mode)
        return start('audit', record, lambda event: target.run_audit(record['id'], cancelled=event.is_set))

    @app.get('/quality/audits/{audit_id}')
    @guarded
    def get_audit(audit_id: str):
        return store().get('audit', audit_id)

    @app.get('/quality/audits/{audit_id}/findings')
    @guarded
    def findings(audit_id: str, offset: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=500)):
        return store().findings(audit_id, offset, limit)

    def cancel(kind, record_id):
        with lock:
            if record_id in jobs:
                jobs[record_id].set()
        return store().cancel(kind, record_id)

    @app.post('/quality/audits/{audit_id}/cancel')
    @guarded
    def cancel_audit(audit_id: str):
        return cancel('audit', audit_id)

    @app.post('/quality/audits/{audit_id}/review')
    @guarded
    def review(audit_id: str, body: ReviewRequest):
        return store().review(audit_id, body.fingerprint, body.decisions, body.exclusions)

    @app.post('/quality/audits/{audit_id}/versions')
    @guarded
    def version(audit_id: str, body: VersionRequest):
        return store().materialize(audit_id, body.fingerprint)

    @app.get('/quality/versions')
    @guarded
    def versions():
        return store().list('version')

    @app.get('/quality/versions/{version_id}/export')
    @guarded
    def export_version(version_id: str, kind: str = 'dataset'):
        target = store()
        record = target.get('version', version_id)
        path = target.export_path(version_id)
        if kind == 'dataset':
            return FileResponse(path, media_type='application/x-ndjson', filename=f'{version_id}.jsonl')
        if kind != 'report':
            raise ValueError('Export kind must be dataset or report.')

        def report():
            yield encoded({'version': record}) + '\n'
            offset = 0
            while True:
                page = target.version_findings(version_id, offset, 100)
                for item in page['items']:
                    yield encoded({'finding': item}) + '\n'
                offset += len(page['items'])
                if offset >= page['total']:
                    break
        return StreamingResponse(report(), media_type='application/x-ndjson', headers={'Content-Disposition': f'attachment; filename="{version_id}-report.jsonl"'})

    @app.get('/quality/evaluations')
    @guarded
    def evaluations():
        return store().list('evaluation')

    @app.post('/quality/evaluations', status_code=202)
    @guarded
    def create_evaluation(body: EvaluationRequest):
        from quality_evaluation import evaluate
        policy = normalize_policy(body.policy)
        target = store()
        cases = body.cases
        benchmark = None
        if body.benchmarkRevision:
            if cases:
                raise ValueError('Benchmark cases are immutable; do not supply replacement cases.')
            from quality_models import load_benchmark
            benchmark = load_benchmark(body.benchmarkRevision, runtime_root=target.root, all_subjects=body.allSubjects)
            cases = benchmark['cases']
        if not cases:
            raise ValueError('Provide frozen cases or a prepared benchmark revision.')
        if body.generate and body.answers:
            raise ValueError('Choose uploaded answers or local generation, not both.')
        evaluate(cases, [], policy)  # Validate frozen cases before any expensive model work.
        adapter = None
        if body.adapterRunId:
            if not body.generate:
                raise ValueError('An adapter requires local generation.')
            run_dir = (target.root / 'runs' / identifier(body.adapterRunId)).resolve()
            if not run_dir.is_relative_to(target.root):
                raise ValueError('Invalid local adapter reference.')
            run = json.loads((run_dir / 'run.json').read_text(encoding='utf-8'))
            if run.get('status') != 'complete':
                raise ValueError('Adapter run is not complete.')
            adapter = (run_dir / 'adapter').resolve()
            if not adapter.is_relative_to(run_dir):
                raise ValueError('Invalid adapter artifact.')
        record = {'loadMode': body.loadMode, 'id': str(uuid.uuid4()), 'createdAt': datetime.now(timezone.utc).isoformat(), 'status': 'queued', 'policy': policy,
                  'frozenCasesHash': digest(cases), 'cases': cases, 'benchmark': {k: v for k, v in benchmark.items() if k != 'cases'} if benchmark else None}
        target.put('evaluation', record)

        def run(event):
            try:
                record['status'] = 'running'
                target.put('evaluation', record)
                answers = body.answers
                if body.generate:
                    from quality_pilot import generate_bounded
                    generated = generate_bounded(cases, runtime_root=target.root, adapter_path=adapter, cancelled=event.is_set, max_seconds=body.maxGpuSeconds, **({"load_mode": body.loadMode} if body.loadMode != "fp16" else {}))
                    record['generation'] = generated
                    if generated['status'] != 'complete':
                        record.update(status=generated['status'], message=generated['message'])
                        target.put('evaluation', record)
                        return
                    answers = generated['answers']
                    from quality_evaluation import fingerprint as policy_fingerprint
                    for answer in answers:
                        answer['policyHash'] = policy_fingerprint(policy)
                record['result'] = evaluate(cases, answers, policy, cancelled=event.is_set)
                record['status'] = 'cancelled' if event.is_set() else 'complete'
            except InterruptedError:
                record['status'] = 'cancelled'
            except ValueError as error:
                record.update(status='failed', message=str(error))
            except Exception:
                record.update(status='failed', message='Local evaluation failed. Check storage and installed quality resources.')
            target.put('evaluation', record)
        return start('evaluation', record, run)

    @app.get('/quality/evaluations/{evaluation_id}')
    @guarded
    def get_evaluation(evaluation_id: str):
        return store().get('evaluation', evaluation_id)

    @app.post('/quality/evaluations/{evaluation_id}/cancel')
    @guarded
    def cancel_evaluation(evaluation_id: str):
        return cancel('evaluation', evaluation_id)

    @app.get('/quality/evaluations/{evaluation_id}/export')
    @guarded
    def export_evaluation(evaluation_id: str):
        record = store().get('evaluation', evaluation_id)
        return StreamingResponse(iter([encoded(record)]), media_type='application/json', headers={'Content-Disposition': f'attachment; filename="{record["id"]}-evaluation.json"'})
