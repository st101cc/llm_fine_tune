"""Conservative, review-only zh-TW suggestions; no model or network calls."""
import hashlib
import json
import re
from difflib import SequenceMatcher
from functools import lru_cache
from importlib.metadata import version

VERSION = 'zh-TW-review-v1'
OPENCC_VERSION = '1.4.2'
PROTECTED = re.compile(r'(?P<fence>`{3,}|~{3,})[^\n]*\n[\s\S]*?(?:^[ \t]*(?P=fence)[ \t]*$|\Z)|(?P<ticks>`+)(?!`)[\s\S]*?(?<!`)(?P=ticks)(?!`)|^(?: {4}|\t).*|https?://[^\s<>]+|「[^」]*」|『[^』]*』|“[^”]*”|‘[^’]*’|"(?:\\.|[^"\\])*"|\'[^\'\n]*\'', re.M)
AMBIGUOUS = set('干面后里云台只斗余系松征谷范丑制复划冲于周布表向舍秋叶咸准出借回困才折据朱术朴杰板极柳核梁棉沈注汇汪泛泪洒淀游溪烟琅瓜画皂碱种秘群床峰粽庄晒痴霉虱灶昵肴羡痒')


def normalize_policy(policy=None):
    policy = policy or {}
    if not isinstance(policy, dict) or set(policy) - {'version', 'terms', 'exclusions', 'fields', 'id', 'openccVersion'}:
        raise ValueError('Invalid quality policy fields.')
    if policy.get('version', VERSION) != VERSION or policy.get('openccVersion', OPENCC_VERSION) != OPENCC_VERSION:
        raise ValueError('Unsupported quality policy or OpenCC version; prepare the recorded resources instead of relabeling it.')
    terms, exclusions, fields = policy.get('terms', {}), policy.get('exclusions', []), policy.get('fields', [])
    if not isinstance(terms, dict) or len(terms) > 1000 or any(not isinstance(k, str) or not k or len(k) > 200 or not isinstance(v, str) or not v or len(v) > 200 for k, v in terms.items()):
        raise ValueError('Terms require nonempty text: at most 1000 entries, 200 characters each.')
    if not isinstance(exclusions, list) or len(exclusions) > 1000 or any(not isinstance(s, str) or not s or len(s) > 500 for s in exclusions):
        raise ValueError('Exclusions require nonempty text: at most 1000 entries, 500 characters each.')
    if not isinstance(fields, list) or any(x != 'text' for x in fields):
        raise ValueError('Only raw text requires explicit selection: ["text"].')
    result = {'version': VERSION, 'openccVersion': OPENCC_VERSION, 'terms': dict(sorted(terms.items())), 'exclusions': sorted(set(exclusions)), 'fields': sorted(set(fields))}
    result['id'] = hashlib.sha256(json.dumps(result, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    return result


@lru_cache(maxsize=1)
def converters():
    try:
        if version('OpenCC') != OPENCC_VERSION:
            raise ValueError('OpenCC version mismatch.')
        from opencc import OpenCC
        return OpenCC('s2t.json'), OpenCC('s2twp.json'), OpenCC('t2s.json')
    except Exception as error:
        raise ValueError(f'Offline quality resources unavailable. Install OpenCC=={OPENCC_VERSION} during setup.') from error


def inspect_text(text, policy=None, *, cancelled=lambda: False):
    if not isinstance(text, str):
        raise ValueError('Inspection requires text.')
    policy = normalize_policy(policy)
    if cancelled():
        raise InterruptedError()
    basic, taiwan, simplified = converters()
    protected = [(m.start(), m.end()) for m in PROTECTED.finditer(text)]
    for exclusion in policy['exclusions']:
        protected.extend((m.start(), m.end()) for m in re.finditer(re.escape(exclusion), text))
    mask = bytearray(len(text))
    for start, end in protected:
        mask[start:end] = b'\x01' * (end - start)
    def overlaps(start, end):
        return mask.find(b'\x01', start, end) != -1
    findings = []
    def add(start, end, replacement, code, reason):
        if start == end or overlaps(start, end):
            return
        findings.append({'code': code, 'start': start, 'end': end, 'original': text[start:end], 'replacement': replacement, 'reason': reason, 'editable': True})
        protected.append((start, end))
        mask[start:end] = b'\x01' * (end - start)
    for term, replacement in sorted(policy['terms'].items(), key=lambda pair: -len(pair[0])):
        if cancelled():
            raise InterruptedError()
        if term != replacement:
            for match in re.finditer(re.escape(term), text):
                add(match.start(), match.end(), replacement, 'terminology', 'Customer terminology preference; review context.')
    protected.extend((i, i + 1) for i, ch in enumerate(text) if ch in AMBIGUOUS)
    for i, ch in enumerate(text):
        if ch in AMBIGUOUS:
            mask[i] = 1
    # Bound each diff, including adversarial repetitive content. Boundary-spanning
    # terminology may need a customer mapping; never trade safety for recall.
    boundaries = sorted(set(range(0, len(text), 256)) | {0, len(text)} | {n for span in protected for n in span})
    has_simplified = has_traditional = False
    for left, right in zip(boundaries, boundaries[1:]):
        if cancelled():
            raise InterruptedError()
        if overlaps(left, right):
            continue
        source = text[left:right]
        has_simplified |= any(ch not in AMBIGUOUS and basic.convert(ch) != ch for ch in source)
        has_traditional |= any(ch not in AMBIGUOUS and simplified.convert(ch) != ch for ch in source)
        target = taiwan.convert(source)
        for op, a, b, c, d in SequenceMatcher(None, source, target, autojunk=True).get_opcodes():
            if op == 'equal' or not source[a:b] or all(ch in AMBIGUOUS for ch in source[a:b]):
                continue
            kind = 'script_candidate' if basic.convert(source[a:b]) != source[a:b] else 'terminology'
            add(left + a, left + b, target[c:d], kind, 'OpenCC candidate, not proof of an error; review meaning and names.')
    if has_simplified and has_traditional:
        findings.append({'code': 'mixed_script', 'start': 0, 'end': len(text), 'original': '', 'replacement': None, 'reason': 'Both script indicators occur; intentional mixtures are valid.', 'editable': False})
    return sorted(findings, key=lambda f: (f['start'], f['end']))


def text_fields(row, policy=None):
    policy = normalize_policy(policy)
    if isinstance(row.get('messages'), list):
        messages = row['messages']
        return [(['messages', i, 'content'], m['content'], i == len(messages) - 1 and m.get('role') == 'assistant') for i, m in enumerate(messages) if isinstance(m, dict) and isinstance(m.get('content'), str)]
    if 'prompt' in row or 'completion' in row:
        return [([key], row[key], key == 'completion') for key in ('prompt', 'completion') if isinstance(row.get(key), str)]
    if isinstance(row.get('text'), str):
        return [(['text'], row['text'], 'text' in policy['fields'] and not any(k in row for k in ('label', 'labels', 'category', 'target')))]
    return []


def row_problem(row):
    if not isinstance(row, dict):
        return 'Row must be an object.'
    if 'messages' in row:
        messages = row['messages']
        if not isinstance(messages, list) or not messages or any(not isinstance(m, dict) or m.get('role') not in ('system', 'user', 'assistant') or not isinstance(m.get('content'), str) or not m['content'].strip() for m in messages):
            return 'Messages require supported roles and nonempty text.'
        if messages[-1]['role'] != 'assistant':
            return 'Chat training example has no final assistant answer.'
    elif 'prompt' in row or 'completion' in row:
        if any(not isinstance(row.get(k), str) or not row[k].strip() for k in ('prompt', 'completion')):
            return 'Prompt and completion must both be nonempty text.'
    elif not isinstance(row.get('text'), str) or not row['text'].strip():
        return 'Expected messages, prompt/completion, or nonempty text.'
    return None
