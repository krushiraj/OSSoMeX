"""Public registry source URLs and parsers into one normalized record shape."""

from urllib.parse import urlencode

from .normalize import package_id, package_ids_from_url, repo_key

ECOSYSTEMS_API = 'https://papers.ecosyste.ms/api/v1'
BIOTOOLS_API = 'https://bio.tools/api/tool/'
WIKIDATA_SPARQL = 'https://query.wikidata.org/sparql'
LICENCES = {'ecosystems': 'CC-BY-SA-4.0', 'biotools': 'CC-BY-4.0', 'wikidata': 'CC0-1.0'}
SOURCE_PRIORITY = {'wikidata': 0, 'biotools': 1, 'ecosystems': 2}

# Software (Q7397) with a repository or official website; websites keep proprietary tools such as SPSS.
WIKIDATA_LIST_QUERY = ('SELECT DISTINCT ?s WHERE { ?s wdt:P31/wdt:P279* wd:Q7397 . '
                       '{ ?s wdt:P1324 [] } UNION { ?s wdt:P856 [] } '
                       'MINUS { ?s wdt:P31/wdt:P279* wd:Q7889 } }')
WIKIDATA_FIELDS = [('repo', 'wdt:P1324'), ('website', 'wdt:P856'), ('pypi', 'wdt:P5568'), ('cran', 'wdt:P5565'),
                   ('bioconductor', 'wdt:P10892'), ('rrid', 'wdt:P9712'), ('doi', 'wdt:P1343/wdt:P356'),
                   ('instance_of', 'wdt:P31'), ('sitelinks', 'wikibase:sitelinks')]


def ecosystems_page_url(page):
    return f'{ECOSYSTEMS_API}/projects?' + urlencode({'per_page': 100, 'page': page})


def biotools_page_url(page):
    return BIOTOOLS_API + '?' + urlencode({'format': 'json', 'per_page': 100, 'page': page})


def _sparql_url(query):
    return WIKIDATA_SPARQL + '?' + urlencode({'query': query, 'format': 'json'})


def wikidata_list_url():
    return _sparql_url(WIKIDATA_LIST_QUERY)


def wikidata_layer_list_url(layer):
    """Optional extra classes (e.g. programming languages) whose items have one of the required properties."""
    # One UNION branch per class: a VALUES list over the subclass path times out on the public endpoint.
    classes = ' UNION '.join(f'{{ ?s wdt:P31/wdt:P279* wd:{c} }}' for c in layer['classes'])
    required = ' UNION '.join(f'{{ ?s wdt:{p} [] }}' for p in layer['require'])
    return _sparql_url(f'SELECT DISTINCT ?s WHERE {{ {classes} {required} }}')


def wikidata_detail_url(qids):
    # Many items keep their label only under the language-neutral "mul" code.
    unions = [f'{{ ?s rdfs:label ?v FILTER(LANG(?v) IN ("en", "mul")) BIND("label" AS ?k) }}',
              f'{{ ?s skos:altLabel ?v FILTER(LANG(?v) IN ("en", "mul")) BIND("alias" AS ?k) }}',
              f'{{ ?s schema:description ?v FILTER(LANG(?v)="en") BIND("description" AS ?k) }}']
    unions += [f'{{ ?s {path} ?v BIND("{key}" AS ?k) }}' for key, path in WIKIDATA_FIELDS]
    values = ' '.join('wd:' + q for q in qids)
    return _sparql_url(f'SELECT ?s ?k ?v WHERE {{ VALUES ?s {{ {values} }} {" UNION ".join(unions)} }}')


def qid(uri):
    return uri.rsplit('/', 1)[-1]


def _unique(values):
    return list(dict.fromkeys(v for v in values if v))


def _record(source, source_id, name, *, aliases=(), description=None, homepages=(), repo_urls=(),
            package_ids=None, rrids=(), dois=(), topics=(), classes=(), czi=None, sitelinks=0):
    ids = {k: _unique(v) for k, v in (package_ids or {}).items()}
    for url in list(homepages) + list(repo_urls):
        found = package_ids_from_url(url)
        if found:
            ids.setdefault(found[0], [])
            ids[found[0]] = _unique(ids[found[0]] + [found[1]])
    row = {'source': source, 'source_id': source_id, 'key': f'{source}:{source_id}', 'name': name or None,
           'aliases': _unique(a for a in aliases if a != name), 'description': (description or '').strip() or None,
           'homepages': _unique(homepages), 'repo_urls': _unique(repo_urls),
           'repo_keys': _unique(repo_key(u) for u in list(repo_urls) + list(homepages)),
           'package_ids': {k: v for k, v in sorted(ids.items()) if v},
           'rrids': _unique(r.strip().upper().removeprefix('RRID:') for r in rrids if r),
           'publication_dois': _unique(d.strip().lower() for d in dois if d), 'topics': _unique(topics),
           'classes': _unique(classes), 'sitelinks': sitelinks}
    if czi is not None:
        row['czi'] = czi
    return row


def parse_ecosystems(row):
    package = row.get('package') or {}
    ecosystem, name = row['ecosystem'], package.get('name') or row['name']
    return _record('ecosystems', f"{ecosystem}/{row['name']}", name, description=package.get('description'),
                   homepages=[package.get('homepage')], repo_urls=[package.get('repository_url')],
                   package_ids={ecosystem: [package_id(ecosystem, name)]}, topics=package.get('keywords_array') or [],
                   czi={'czi_id': row['czi_id'], 'czi_name': row['name'], 'mentions_count': row['mentions_count'],
                        'project_url': row['project_url']})


def parse_biotools(tool):
    repos = [l['url'] for l in tool.get('link') or [] if 'Repository' in (l.get('type') or [])]
    repos += [d['url'] for d in tool.get('download') or [] if d.get('type') == 'Source code']
    return _record('biotools', tool['biotoolsID'], tool.get('name'), description=tool.get('description'),
                   homepages=[tool.get('homepage')], repo_urls=repos,
                   rrids=[o.get('value') for o in tool.get('otherID') or [] if (o.get('type') or '').lower() == 'rrid'],
                   dois=[p.get('doi') for p in tool.get('publication') or []],
                   topics=[t.get('term') for t in tool.get('topic') or []])


def parse_wikidata(bindings):
    grouped = {}
    for b in sorted(bindings, key=lambda b: b['v'].get('xml:lang') != 'en'):
        grouped.setdefault(qid(b['s']['value']), {}).setdefault(b['k']['value'], []).append(b['v']['value'])
    records = []
    for item, fields in sorted(grouped.items()):
        get = lambda key: fields.get(key, [])
        records.append(_record(
            'wikidata', item, (get('label') or [None])[0], aliases=get('label')[1:] + get('alias'),
            description=(get('description') or [None])[0], homepages=get('website'), repo_urls=get('repo'),
            package_ids={e: [package_id(e, n) for n in get(e)] for e in ('pypi', 'cran', 'bioconductor')},
            rrids=get('rrid'), dois=get('doi'), classes=[qid(c) for c in get('instance_of')],
            sitelinks=max((int(v) for v in get('sitelinks')), default=0)))
    return records
