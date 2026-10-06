"""Local audit persistence and streaming, immutable dataset derivatives."""
import csv
import hashlib
import json
import os
import random
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from quality_policy import inspect_text, normalize_policy, row_problem, text_fields


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def digest(value):
    return hashlib.sha256(encoded(value).encode('utf-8')).hexdigest()


def fingerprint(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def identifier(value):
    try:
        return str(uuid.UUID(str(value)))
    except (ValueError, TypeError, AttributeError) as error:
        raise ValueError('Invalid record identifier.') from error


def iter_rows(path):
    path = Path(path)
    if path.suffix.lower() == '.jsonl':
        with path.open(encoding='utf-8-sig') as stream:
            for line in stream:
                try:
                    yield json.loads(line, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
                except ValueError:
                    yield {'__quality_invalid_json__': True}
    elif path.suffix.lower() == '.csv':
        with path.open(encoding='utf-8-sig', newline='') as stream:
            yield from csv.DictReader(stream)
    elif path.suffix.lower() == '.parquet':
        import pyarrow.parquet as parquet
        for batch in parquet.ParquetFile(path).iter_batches(batch_size=128):
            yield from batch.to_pylist()
    else:
        raise ValueError('Supported local datasets: JSONL, CSV, Parquet.')


class QualityStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.datasets = self.root / 'datasets'
        self.directory = self.root / 'quality'
        self.datasets.mkdir(parents=True, exist_ok=True)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.database = self.directory / 'quality.sqlite'
        with self.connection() as db:
            db.executescript('''
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS records(kind TEXT, id TEXT PRIMARY KEY, payload TEXT);
                CREATE INDEX IF NOT EXISTS record_kind ON records(kind);
                CREATE TABLE IF NOT EXISTS findings(seq INTEGER PRIMARY KEY, id TEXT UNIQUE,
                    audit TEXT, row_id INTEGER, body TEXT, decision TEXT DEFAULT 'pending');
                CREATE INDEX IF NOT EXISTS finding_audit ON findings(audit, row_id, seq);
                CREATE TABLE IF NOT EXISTS seen(audit TEXT, kind TEXT, key TEXT, row_id INTEGER,
                    answer TEXT, PRIMARY KEY(audit,kind,key));
                CREATE TABLE IF NOT EXISTS exceptions(policy TEXT, term TEXT, PRIMARY KEY(policy,term));
                CREATE TABLE IF NOT EXISTS version_findings(version TEXT, seq INTEGER, body TEXT, decision TEXT, PRIMARY KEY(version,seq));
                CREATE TABLE IF NOT EXISTS review_history(seq INTEGER PRIMARY KEY, audit TEXT, finding TEXT, decision TEXT, created TEXT);
            ''')

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.database, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def put(self, kind, record, db=None):
        if db is None:
            with self.connection() as connection:
                self.put(kind, record, connection)
            return record
        db.execute('INSERT INTO records VALUES(?,?,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload', (kind, identifier(record['id']), encoded(record)))
        return record

    def get(self, kind, record_id):
        with self.connection() as db:
            row = db.execute('SELECT payload FROM records WHERE kind=? AND id=?', (kind, identifier(record_id))).fetchone()
        if row is None:
            raise KeyError('Record not found.')
        return json.loads(row[0])

    def list(self, kind, limit=100):
        with self.connection() as db:
            return [json.loads(r[0]) for r in db.execute('SELECT payload FROM records WHERE kind=? ORDER BY rowid DESC LIMIT ?', (kind, limit))]

    def recover(self):
        with self.connection() as db:
            # A crash while writing (before publication intent) can leave a
            # server-named partial. Preserve it in quarantine so retries work.
            for partial in self.datasets.glob('*.jsonl.partial'):
                try:
                    version_id = identifier(partial.name.removesuffix('.jsonl.partial'))
                except ValueError:
                    continue
                if not partial.resolve().is_relative_to(self.datasets.resolve()):
                    continue
                if db.execute('SELECT 1 FROM records WHERE kind=? AND id=?', ('version', version_id)).fetchone():
                    continue
                quarantine = self.directory / 'recovery-orphans'
                quarantine.mkdir(exist_ok=True)
                os.replace(partial, quarantine / (version_id + '-' + str(uuid.uuid4()) + '.partial'))
            for row in db.execute('SELECT kind,payload FROM records').fetchall():
                record = json.loads(row['payload'])
                if row['kind'] == 'version' and record.get('status') == 'publishing':
                    destination = self.datasets / (identifier(record['id']) + '.jsonl')
                    partial = destination.with_suffix('.jsonl.partial')
                    expected = record['provenance']['outputFingerprint']
                    try:
                        if not destination.exists() and partial.is_file() and fingerprint(partial) == expected:
                            os.replace(partial, destination)
                        if destination.is_file() and fingerprint(destination) == expected:
                            record['status'] = 'complete'
                        else:
                            record.update(status='interrupted', message='Version publication incomplete; original preserved. Create a new audit.')
                        self.put('version', record, db)
                    except OSError:
                        pass  # Preserve durable intent for the next storage recovery.
                if record.get('status') in ('queued', 'running', 'cancelling'):
                    record.update(status='interrupted', message='Interrupted by service restart; start a new job. Completed artifacts were preserved.')
                    self.put(row['kind'], record, db)

    def source_path(self, ref):
        if ref.get('source') != 'upload':
            raise ValueError('Import public data first; quality processing reads local files only.')
        path = Path(ref.get('name', '')).resolve()
        if not path.is_relative_to(self.datasets.resolve()) or not path.is_file() or path.suffix.lower() not in ('.jsonl', '.csv', '.parquet'):
            raise ValueError('Dataset must be a local imported file in the dataset storage area.')
        return path

    def create_audit(self, dataset, policy=None, mode='full'):
        if mode not in ('full', 'sample'):
            raise ValueError('Audit mode must be full or sample.')
        path = self.source_path(dataset)
        policy = normalize_policy(policy)
        with self.connection() as db:
            saved = [r[0] for r in db.execute('SELECT term FROM exceptions WHERE policy=?', (policy['id'],))]
        if saved:
            policy = normalize_policy({**policy, 'exclusions': policy['exclusions'] + saved})
        record = {'id': str(uuid.uuid4()), 'createdAt': datetime.now(timezone.utc).isoformat(), 'status': 'queued',
                  'dataset': {**dataset, 'name': str(path)}, 'policy': policy, 'mode': mode, 'coverage': 'pending',
                  'scannedRows': 0, 'totalRows': None, 'findingCount': 0, 'skippedFields': 0, 'fingerprint': None}
        return self.put('audit', record)

    def cancel(self, kind, record_id):
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT payload FROM records WHERE kind=? AND id=?', (kind, identifier(record_id))).fetchone()
            if row is None:
                raise KeyError('Record not found.')
            record = json.loads(row[0])
            if record['status'] in ('queued', 'running'):
                record['status'] = 'cancelling'
                self.put(kind, record, db)
        return record

    def _finding(self, db, audit, row_id, body):
        body = {'field': [], 'start': 0, 'end': 0, 'original': '', 'replacement': None, 'editable': False, **body,
                'id': str(uuid.uuid4()), 'rowId': row_id}
        db.execute('INSERT INTO findings(id,audit,row_id,body) VALUES(?,?,?,?)', (body['id'], audit, row_id, encoded(body)))

    def run_audit(self, audit_id, cancelled=lambda: False):
        record = self.get('audit', audit_id)
        if record['status'] not in ('queued', 'cancelling'):
            raise ValueError('An audit may only be executed once.')
        def stopped():
            return cancelled() or self.get('audit', audit_id)['status'] == 'cancelling'
        try:
            if stopped():
                raise InterruptedError()
            record['status'] = 'running'
            self.put('audit', record)
            path = self.source_path(record['dataset'])
            record['fingerprint'] = fingerprint(path)
            total = 0
            for total, _ in enumerate(iter_rows(path), 1):
                if total % 100 == 0 and stopped():
                    raise InterruptedError()
            record['totalRows'] = total
            selected = set(random.Random(42).sample(range(total), min(total, 600))) if record['mode'] == 'sample' else None
            self.put('audit', record)
            with self.connection() as db:
                for row_id, row in enumerate(iter_rows(path)):
                    if row_id % 50 == 0:
                        db.commit()
                        if stopped():
                            raise InterruptedError()
                    if selected is not None and row_id not in selected:
                        continue
                    problem = row_problem(row)
                    if problem:
                        self._finding(db, audit_id, row_id, {'code': 'malformed', 'reason': problem})
                    if isinstance(row, dict):
                        key = digest(row)
                        previous = db.execute('SELECT row_id FROM seen WHERE audit=? AND kind=? AND key=?', (audit_id, 'row', key)).fetchone()
                        if previous:
                            self._finding(db, audit_id, row_id, {'code': 'duplicate', 'reason': f'Exact duplicate of row {previous[0]}.'})
                        db.execute('INSERT OR IGNORE INTO seen VALUES(?,?,?,?,?)', (audit_id, 'row', key, row_id, ''))
                        if not problem:
                            from advisor import input_key, classification_label_field
                            input_hash = input_key(row)
                            label = classification_label_field(list(row))
                            answer = row.get('completion', row['messages'][-1]['content'] if 'messages' in row else row.get(label) if label else None)
                            if answer is not None:
                                answer_hash = digest(answer)
                                earlier = db.execute('SELECT row_id,answer FROM seen WHERE audit=? AND kind=? AND key=?', (audit_id, 'input', input_hash)).fetchone()
                                if earlier and earlier['answer'] != answer_hash:
                                    self._finding(db, audit_id, row_id, {'code': 'conflicting_answer', 'reason': f'Same normalized input as row {earlier["row_id"]}, different answer; review rather than delete.'})
                                db.execute('INSERT OR IGNORE INTO seen VALUES(?,?,?,?,?)', (audit_id, 'input', input_hash, row_id, answer_hash))
                        for field, text, editable in text_fields(row, record['policy']):
                            if len(text) > 65536:
                                record['skippedFields'] += 1
                                self._finding(db, audit_id, row_id, {'code': 'field_too_large', 'field': field, 'reason': 'Field exceeds 65536 characters; linguistic inspection skipped, not passed.'})
                                continue
                            for finding in inspect_text(text, record['policy'], cancelled=cancelled):
                                self._finding(db, audit_id, row_id, {**finding, 'field': field, 'editable': editable and finding['editable']})
                    record['scannedRows'] += 1
                    if record['scannedRows'] % 50 == 0:
                        self.put('audit', record, db)
                record['findingCount'] = db.execute('SELECT count(*) FROM findings WHERE audit=?', (audit_id,)).fetchone()[0]
            if stopped():
                raise InterruptedError()
            if fingerprint(path) != record['fingerprint']:
                raise ValueError('Dataset changed during audit; start a new audit.')
            record.update(status='complete', coverage='partial' if record['skippedFields'] else record['mode'])
        except InterruptedError:
            record.update(status='cancelled', coverage='partial')
        except Exception as error:
            record.update(status='failed', coverage='partial', message='Audit failed. Verify local resource installation, dataset format and available disk space.', errorType=type(error).__name__)
        self.put('audit', record)
        return record

    def findings(self, audit_id, offset=0, limit=100):
        self.get('audit', audit_id)
        if not 0 <= offset or not 1 <= limit <= 500:
            raise ValueError('Invalid findings page.')
        with self.connection() as db:
            rows = db.execute('SELECT body,decision FROM findings WHERE audit=? ORDER BY row_id,seq LIMIT ? OFFSET ?', (audit_id, limit, offset)).fetchall()
            total = db.execute('SELECT count(*) FROM findings WHERE audit=?', (audit_id,)).fetchone()[0]
        return {'items': [{**json.loads(r['body']), 'decision': r['decision']} for r in rows], 'total': total, 'offset': offset}

    def checked(self, audit_id, expected):
        audit = self.get('audit', audit_id)
        if audit['status'] != 'complete' or audit['coverage'] != 'full':
            raise ValueError('Review and correction require a completed full audit without skipped fields.')
        path = self.source_path(audit['dataset'])
        if expected != audit['fingerprint'] or fingerprint(path) != expected:
            raise ValueError('Dataset changed or review is stale. Start a new audit.')
        return audit, path

    def review(self, audit_id, expected, decisions, exclusions=None):
        audit, _ = self.checked(audit_id, expected)
        if len(decisions) > 500:
            raise ValueError('Review at most 500 findings per request.')
        if len({d.get('id') for d in decisions}) != len(decisions):
            raise ValueError('Duplicate review decisions.')
        saved = normalize_policy({**audit['policy'], 'exclusions': exclusions or []})['exclusions']
        with self.connection() as db:
            for decision in decisions:
                if decision.get('decision') not in ('accepted', 'ignored', 'pending'):
                    raise ValueError('Unknown review decision.')
                row = db.execute('SELECT body FROM findings WHERE audit=? AND id=?', (audit_id, identifier(decision.get('id')))).fetchone()
                if not row or (decision['decision'] == 'accepted' and not json.loads(row[0])['editable']):
                    raise ValueError('Finding is unknown or cannot be corrected.')
                db.execute('UPDATE findings SET decision=? WHERE audit=? AND id=?', (decision['decision'], audit_id, decision['id']))
                db.execute('INSERT INTO review_history(audit,finding,decision,created) VALUES(?,?,?,?)', (audit_id, decision['id'], decision['decision'], datetime.now(timezone.utc).isoformat()))
            for term in saved:
                db.execute('INSERT OR IGNORE INTO exceptions VALUES(?,?)', (audit['policy']['id'], term))
        return {'saved': len(decisions), 'exceptionsSaved': saved, 'message': 'Exceptions apply to new audits; existing evidence is unchanged.'}

    def materialize(self, audit_id, expected):
        audit, path = self.checked(audit_id, expected)
        with self.connection() as db:
            signature = hashlib.sha256()
            db.execute('BEGIN IMMEDIATE')
            for row in db.execute('SELECT id,decision FROM findings WHERE audit=? ORDER BY seq', (audit_id,)):
                signature.update(encoded(list(row)).encode())
            version_id = str(uuid.uuid5(uuid.UUID(audit_id), signature.hexdigest()))
            existing = db.execute('SELECT payload FROM records WHERE kind=? AND id=?', ('version', version_id)).fetchone()
            if existing:
                result = json.loads(existing[0])
                if result.get('status') != 'complete':
                    raise ValueError('Version publication incomplete; restart the service to recover local artifacts.')
                return result
            destination = self.datasets / f'{version_id}.jsonl'
            partial = destination.with_suffix('.jsonl.partial')
            count = edits = 0
            owned = False
            try:
                with partial.open('x', encoding='utf-8', newline='\n') as stream:
                    owned = True
                    for row_id, row in enumerate(iter_rows(path)):
                        if not isinstance(row, dict) or row.get('__quality_invalid_json__'):
                            raise ValueError('Malformed JSON cannot be exported as a corrected version.')
                        changes = [json.loads(r[0]) for r in db.execute('SELECT body FROM findings WHERE audit=? AND row_id=? AND decision=? ORDER BY seq', (audit_id, row_id, 'accepted'))]
                        by_field = {}
                        for change in changes:
                            by_field.setdefault(tuple(change['field']), []).append(change)
                        for field, group in by_field.items():
                            parent = row
                            for key in field[:-1]:
                                parent = parent[key]
                            text = parent[field[-1]]
                            end = 0
                            for change in sorted(group, key=lambda c: c['start']):
                                if change['start'] < end or text[change['start']:change['end']] != change['original']:
                                    raise ValueError('Stale or overlapping corrections rejected.')
                                end = change['end']
                            for change in sorted(group, key=lambda c: c['start'], reverse=True):
                                text = text[:change['start']] + change['replacement'] + text[change['end']:]
                                edits += 1
                            parent[field[-1]] = text
                        stream.write(encoded(row) + '\n')
                        count += 1
                if fingerprint(path) != expected:
                    raise ValueError('Dataset changed during correction.')
                if destination.exists():
                    raise ValueError('Version file already exists without a complete record; start a new audit.')
                result = {'id': version_id, 'auditId': audit_id, 'rows': count, 'edits': edits,
                          'status': 'publishing',
                          'dataset': {'source': 'upload', 'name': str(destination), 'extension': 'jsonl', 'displayName': f'Reviewed dataset {version_id[:8]}', 'qualityVersionId': version_id},
                          'provenance': {'sourceFingerprint': expected, 'outputFingerprint': fingerprint(partial), 'policy': audit['policy'], 'auditId': audit_id, 'reviewDigest': signature.hexdigest(), 'rowLineage': 'Original zero-based row indices preserved; no rows reordered or removed.'}}
                db.execute('INSERT INTO version_findings SELECT ?,seq,body,decision FROM findings WHERE audit=?', (version_id, audit_id))
                self.put('version', result, db)
                db.commit()
                # The durable intent lets startup reconcile a crash before or
                # after publication; no existing original is overwritten.
                owned = False
                os.replace(partial, destination)
                result['status'] = 'complete'
                self.put('version', result, db)
                return result
            finally:
                if owned and partial.exists():
                    partial.unlink()

    def version_findings(self, version_id, offset=0, limit=100):
        self.get('version', version_id)
        if offset < 0 or not 1 <= limit <= 500:
            raise ValueError('Invalid pagination.')
        with self.connection() as db:
            total = db.execute('SELECT COUNT(*) FROM version_findings WHERE version=?', (version_id,)).fetchone()[0]
            items = [{**json.loads(r['body']), 'decision': r['decision']} for r in db.execute('SELECT body,decision FROM version_findings WHERE version=? ORDER BY seq LIMIT ? OFFSET ?', (version_id, limit, offset))]
        return {'items': items, 'total': total, 'offset': offset}

    def export_path(self, version_id):
        version = self.get('version', version_id)
        if version.get('status') != 'complete':
            raise ValueError('Version publication is not complete.')
        path = self.source_path(version['dataset'])
        if fingerprint(path) != version['provenance']['outputFingerprint']:
            raise ValueError('Version artifact changed; export refused.')
        return path
