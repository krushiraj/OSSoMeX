"""Build conservative corpus records without inventing access or split evidence."""

from copy import deepcopy
import json
from pathlib import Path
import re

from ..contracts import validate_document
from .acquire import zip_members
from .brat import read_brat
from .tei import read_tei
from .coverage import map_annotations
from .import_bundle import native_policy
from .manifest import digest, fulltext_eligible, json_bytes, read_jsonl, verified_path, write_jsonl, write_once


def native_partition(identifier: str, partitions: dict) -> str:
    roles = [role for role, ids in partitions.items() if identifier in ids]
    if not roles:
        raise ValueError('NATIVE_SPLIT_MISSING')
    if len(roles) != 1:
        raise ValueError('NATIVE_SPLIT_CONFLICT')
    role = {'devel':'dev','validation':'dev'}.get(roles[0], roles[0])
    if role not in ('train', 'dev', 'test'):
        raise ValueError('NATIVE_SPLIT_UNKNOWN')
    return role


def apply_access_decision(document: dict, decision: dict) -> dict:
    if decision.get('document_id') != document['document_id'] or decision.get('text_revision') != document['text_revision']:
        raise ValueError('REVISION_MISMATCH')
    if document.get('supplied_text_scope') in ('methods','sentence') and decision.get('supplied_text_scope') == 'fulltext':
        raise ValueError('EXCERPT_REQUIRES_NEW_TEXT')
    allowed = ('public','language','text_license','access_basis','supplied_text_scope')
    result = {**deepcopy(document), **{k:decision[k] for k in allowed if k in decision}}
    result['access_decision_sha256'] = digest(json_bytes(decision))
    result['fulltext_eligible'] = fulltext_eligible(result)
    return result


def build_corpus(source_manifest: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    rows = read_jsonl(source_manifest)
    for row in rows:
        if 'path' in row:
            verified_path(source_manifest.parent, row)
    native_splits = {}
    for row in rows:
        if row['source'] == 'somesci-split':
            native_splits = json.loads(verified_path(source_manifest.parent, row).read_bytes())
    documents, native, annotations, coverage, issues, normalizations = [], [], [], [], [], []
    for row in rows:
        if row.get('role') not in ('native_archive', 'paper_text'):
            continue
        path = verified_path(source_manifest.parent, row)
        if row['role'] == 'native_archive':
            members = zip_members(path.read_bytes(), row.get('skip_symlinks', False))
            candidates = [(n, b) for n,b in sorted(members.items())
                          if n.endswith('.tei.xml') or n.endswith('.txt') and n[:-4]+'.ann' in members]
        else:
            members = {}
            candidates = [(row['source_record_id'], path.read_bytes())]
        for name, data in candidates:
            is_tei = name.endswith('.tei.xml') or row.get('format') == 'tei'
            fmt = 'tei' if is_tei else 'brat' if row['role'] == 'native_archive' else 'text'
            parsed = (read_tei(data) if is_tei else read_brat(data.decode('utf-8'), members[name[:-4]+'.ann'].decode('utf-8'))
                      if fmt == 'brat' else {'text':data.decode('utf-8'), 'spans':[], 'relations':[], 'issues':[], 'native_records':[]})
            identifier = Path(name).name.removesuffix('.training.tei.xml').removesuffix('.txt')
            if not parsed['text'].strip():
                issues.append({'document_id':row['source']+':'+name, 'status':'input_failed',
                               'code':'EMPTY_SOURCE_TEXT', 'source_member':name})
                continue
            pmc = re.search(r'PMC\d+', identifier)
            partition, partition_issue = None, None
            if row['source'] == 'somesci':
                try:
                    partition = native_partition(pmc.group() if pmc else identifier, native_splits)
                except ValueError as exc:
                    partition_issue = str(exc)
                    partition = 'unknown'
            scope = ('fulltext' if 'Pubmed_fulltext/' in name else 'methods' if 'PLoS_methods/' in name
                     else 'sentence' if '_sentences/' in name else row.get('supplied_text_scope', 'unverified'))
            document = {'document_id': row['source']+':'+name, 'text':parsed['text'], 'source':row['source'],
                        'source_record_id':identifier, 'source_ids':{'pmcid':pmc.group()} if pmc else {},
                        'native_split':partition, 'supplied_text_scope':scope, 'language':None,
                        'public':row.get('public', False), 'text_license':None, 'access_basis':None,
                        'fulltext_eligible':False, 'acquisition_id':row['sha256'],
                        'raw_source':{'path':str(path.resolve()), 'sha256':row['sha256'], 'member':name,
                                      'member_sha256':digest(data), 'release':row.get('release')},
                        'metadata':{'source':row['source'], 'extraction_method':fmt+'-v2',
                                    'domain':name.split('/')[0] if is_tei else None}}
            if row['role'] == 'paper_text':
                document.update({k:row.get(k) for k in ('language','public','text_license','access_basis','supplied_text_scope')})
                document['fulltext_eligible'] = fulltext_eligible(document)
            result = map_annotations(parsed, document, native_policy(fmt, len(parsed['text'])))
            doc = result['document']
            documents.append(doc)
            native.append({'document_id':doc['document_id'], 'original_text_revision':doc['original_text_revision'], **result['native_annotations']})
            annotations.extend(result['occurrences']); coverage.extend(result['coverage'])
            norm = result['normalization']
            normalizations.append({'document_id':doc['document_id'], 'changes':norm['changes'],
                                   'normalizer_version':norm['normalizer_version']})
            issues.append({'document_id':doc['document_id'], 'issues':result['issues'],
                           'exclusions':result['exclusions'], 'partition_issue':partition_issue})
    if not documents:
        raise ValueError('NO_SOURCE_TEXT')
    output.mkdir(parents=True)
    files = {'documents.jsonl':documents, 'native.jsonl':native, 'occurrences.jsonl':annotations,
             'coverage.jsonl':coverage, 'normalization.jsonl':normalizations, 'issues.jsonl':issues}
    for name, values in files.items():
        write_jsonl(output/name, values)
    report = {'status':'built_pending_eligibility', 'document_count':len(documents),
              'input_failed':sum(i.get('status') == 'input_failed' for i in issues),
              'source_manifest':str(source_manifest.resolve()), 'source_manifest_sha256':digest(source_manifest.read_bytes()),
              'sources':[{**r, 'path':str(verified_path(source_manifest.parent,r))} for r in rows if 'path' in r],
              'files':[{'path':n,'sha256':digest((output/n).read_bytes())} for n in files],
              'eligible_fulltext':sum(d['fulltext_eligible'] for d in documents)}
    write_once(output/'manifest.json', json_bytes(report))
    return {k:v for k,v in report.items() if k not in ('sources','files')}


def load_corpus(path: Path) -> list[dict]:
    manifest = json.loads((path/'manifest.json').read_bytes())
    for row in manifest['files']:
        verified_path(path, row)
    for row in manifest['sources']:
        raw = Path(row['path'])
        if not raw.exists() or digest(raw.read_bytes()) != row['sha256']:
            raise ValueError('RAW_SOURCE_MOVED_OR_CHANGED')
    return [validate_document(d) for d in read_jsonl(path/'documents.jsonl')]
