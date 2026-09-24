"""Load versioned annotation rules and bind task identities to exact bytes."""

import re
from pathlib import Path

from ..data.manifest import digest, write_once

ROOT=Path(__file__).resolve().parents[3]


def load_policy(path: Path) -> dict:
    payload=path.read_bytes()
    try:
        content=payload.decode('utf-8')
    except UnicodeDecodeError as exc:
        raise ValueError('INVALID_POLICY_ENCODING') from exc
    headers=re.findall(r'^Version:[ \t]*(\S+)',content,re.MULTILINE)
    if len(headers)!=1 or headers[0].removesuffix('.') not in ('scibert-poc-2.0','scibert-poc-2.1'):
        raise ValueError('POLICY_VERSION_UNSUPPORTED')
    version=headers[0].removesuffix('.')
    if (path.name=='policy.md' and path.parent.name=='scibert-v2' and version!='scibert-poc-2.0'
            or path.name=='policy-2.1.md' and version!='scibert-poc-2.1'):
        raise ValueError('POLICY_VERSION_MISMATCH')
    return {'policy_version':version,'policy_hash':digest(payload),'max_chars':6000,
            'alias_schema_version':'1.0' if version=='scibert-poc-2.1' else None}


def snapshot_prompts(destination: Path, policy: dict) -> dict:
    suffix='-aliases' if policy['alias_schema_version']=='1.0' else ''
    result={}
    for role,stem in (('annotate','annotate'),('check','check')):
        name=f'{stem}{suffix}.md'
        source=ROOT/'prompts'/name
        payload=source.read_bytes()
        relative=f'prompts/{name}'
        write_once(destination/relative,payload)
        result[role]={'path':relative,'sha256':digest(payload),'source_path':str(source)}
    return result
