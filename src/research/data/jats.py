"""Conservative JATS prose extraction and candidate sentence selection."""

import random
import re
import xml.etree.ElementTree as ET
from xml.parsers import expat

from .splits import identifiers

EXTRACTOR_VERSION = 'jats-prose-v1'
SEGMENTER_VERSION = 'sentence-regex-v2'


def normalize_doi(value):
    return next(iter(identifiers({'source_ids':{'doi':value}})), 'doi:').removeprefix('doi:')


def safe_xml(data):
    def reject(*args):
        raise ValueError('UNSAFE_XML')

    guard = expat.ParserCreate()
    guard.StartDoctypeDeclHandler = lambda name, system, public, internal: reject() if internal else None
    guard.EntityDeclHandler = reject
    guard.ExternalEntityRefHandler = reject
    guard.SetParamEntityParsing(expat.XML_PARAM_ENTITY_PARSING_NEVER)
    try:
        guard.Parse(data, True)
        return ET.fromstring(data)
    except (expat.ExpatError, ET.ParseError) as exc:
        raise ValueError('MALFORMED_XML') from exc


def read_jats(data: bytes, *, expected_doi: str, expected_pmcid: str, fallback_language=None) -> dict:
    root = safe_xml(data)
    if root.tag != 'article':
        raise ValueError('NOT_JATS_ARTICLE')
    meta = root.find('./front/article-meta')
    body = root.find('body')
    if meta is None or body is None:
        raise ValueError('MISSING_ARTICLE_BODY')
    ids = {}
    for node in meta.findall('article-id'):
        kind, value = node.get('pub-id-type'), ''.join(node.itertext()).strip()
        if kind in ('doi','pmcid','pmid'):
            if kind == 'doi': value = normalize_doi(value)
            if kind == 'pmcid': value = 'PMC'+value.removeprefix('PMC')
            if ids.get(kind, value) != value: raise ValueError('ARTICLE_IDENTITY_CONFLICT')
            ids[kind] = value
    if ids.get('doi') != normalize_doi(expected_doi) or ids.get('pmcid') != expected_pmcid:
        raise ValueError('ARTICLE_IDENTITY_MISMATCH')
    language = root.get('{http://www.w3.org/XML/1998/namespace}lang') or fallback_language
    if language not in ('en','eng'):
        raise ValueError('NON_ENGLISH_ARTICLE')
    licenses = []
    for node in meta.findall('./permissions/license'):
        content = ' '.join(list(node.itertext())+[v for n in node.iter() for v in n.attrib.values()])
        for match in re.finditer(r'https?://creativecommons\.org/(licenses/(by|by-sa)/(2\.0|3\.0|4\.0)|publicdomain/zero/1\.0)/?(?![\w.-])',content):
            kind = 'CC0-1.0' if match[1].startswith('publicdomain') else 'CC-'+match[2].upper()+'-'+match[3]
            if kind in ('CC-BY-2.0','CC-BY-3.0','CC-BY-4.0','CC-BY-SA-4.0','CC0-1.0'):
                licenses.append((kind,match[0]))
    if not licenses:
        raise ValueError('TEXT_LICENSE_UNSUPPORTED')
    text_of = lambda n: re.sub(r'\s+',' ',''.join(n.itertext())).strip()
    nodes = list(meta.findall('./title-group/article-title'))
    nodes += list(meta.findall('./abstract//p'))
    nodes += list(body.iter('p'))
    paragraphs, parts, offset = [], [], 0
    for index,node in enumerate(nodes):
        # Nested paragraphs (tables/boxed structures) belong to their outer block once.
        if any(node is child for prior in nodes[:index] for child in list(prior.iter())[1:]):
            continue
        value = text_of(node)
        if not value: continue
        if parts: offset += 2
        paragraphs.append({'start':offset,'end':offset+len(value),'xml_block_index':index,'kind':node.tag})
        parts.append(value);offset += len(value)
    if not any(node.tag == 'p' and text_of(node) for node in body.iter('p')):
        raise ValueError('EMPTY_ARTICLE_BODY')
    return {'text':'\n\n'.join(parts), 'paragraphs':paragraphs, 'source_ids':ids, 'language':'en',
            'title':text_of(meta.find('./title-group/article-title')) if meta.find('./title-group/article-title') is not None else None,
            'text_license':licenses[0][0], 'license_url':licenses[0][1],
            'supplied_text_scope':'title_abstract_body_prose', 'fulltext_eligible':False,
            'extractor_version':EXTRACTOR_VERSION, 'normalizer_version':'block-whitespace-v1'}


def sentence_regions(parsed: dict) -> list[dict]:
    text = parsed['text']; result = []
    for block in parsed['paragraphs']:
        if block['kind'] != 'p': continue
        value = text[block['start']:block['end']]; boundaries = [0]
        for m in re.finditer(r'\s+',value):
            preceding = value[:m.start()]
            terminal = re.search(r'[.!?]$',preceding)
            citation = re.search(r'(?<!\d)[.!?]\d+(?:[,–−-]\d+)*$',preceding) and re.match(r'[A-Z]',value[m.end():])
            if not terminal and not citation: continue
            if re.search(r'\b(?:v|vs|e\.g|i\.e|al|Fig|Figs|Dr|Mr|No|eq|ref)\.$',preceding,re.I): continue
            if re.search(r'\b(?:[A-Z]\.){2,}$',preceding): continue
            boundaries.append(m.end())
        boundaries.append(len(value))
        for start,end in zip(boundaries,boundaries[1:]):
            segment = value[start:end].rstrip()
            if not segment: continue
            result.append({'start':block['start']+start,'end':block['start']+start+len(segment),
                           'text':segment,'paragraph_span':{'start':block['start'],'end':block['end']},
                           'segmenter_version':SEGMENTER_VERSION})
    return result


def select_sentences(parsed: dict, aliases: list[str], *, max_matches=4, seed=42) -> list[dict]:
    if type(max_matches) is not int or not 1 <= max_matches <= 20:
        raise ValueError('INVALID_SENTENCE_LIMIT')
    if not aliases or any(not isinstance(a,str) or not a.strip() for a in aliases):
        raise ValueError('INVALID_ALIASES')
    matches, audits = [], []
    for sentence in sentence_regions(parsed):
        hits = [a for a in aliases if re.search(r'(?<!\w)'+re.escape(a)+r'(?!\w)',sentence['text'],0 if len(a)<=3 else re.I)]
        row = {**sentence,'candidate_aliases':hits,'selection_reason':'literal_candidate_match' if hits else 'random_nonmatching_audit'}
        (matches if hits else audits).append(row)
    rng = random.Random(seed)
    selected = sorted(rng.sample(matches,min(max_matches,len(matches))),key=lambda r:r['start'])
    if matches and audits: selected.append(rng.choice(audits))
    return sorted(selected,key=lambda r:r['start'])
