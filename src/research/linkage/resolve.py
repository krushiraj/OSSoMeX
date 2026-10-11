"""Mention-to-entity resolvers: a name-matching baseline and a context-aware resolver that can abstain."""

import math
import re
from collections import Counter

from ..registry.normalize import form_exact, package_ids_from_url, repo_key

STOPWORDS = set('the and for with was were are this that from using used use into its our their which have has been '
                'all also can these than then such each other more most were based analysis data software version'
                .split())
WRAPPER = re.compile(r'\b(wrapper|bindings?|interface (?:to|for)|port of|client for|emulat\w*)\b', re.I)
PLACEHOLDER = re.compile(r'\b(reserved|placeholder|security holding package|dummy package)\b', re.I)
CUES = [('cran', re.compile(r'\bR package|\bCRAN\b|\bin R\b|\bR \(|\bR software|\bRStudio\b|Bioconductor', re.I)),
        ('bioconductor', re.compile(r'Bioconductor', re.I)),
        ('pypi', re.compile(r'\bPython\b|\bpip\b|\bPyPI\b|\bconda\b', re.I))]
MATCH_WEIGHTS = {'name': 2.0, 'alias': 1.8, 'package': 1.6, 'loose': 1.4, 'czi': 1.0, 'token': 0.5, 'url': 0.0}


def _stem(word):
    for suffix in ('ments', 'ment', 'ings', 'ing', 'ers', 'er', 'ed', 'es', 's'):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[:-len(suffix)]
    return word


def tokens(text):
    return [_stem(t) for t in re.findall(r'[a-z][a-z0-9]+', text.casefold()) if t not in STOPWORDS]


def entity_text(entity):
    return ' '.join([*entity['names'], *entity['aliases'], entity['description'] or '', *entity.get('topics', [])])


class FastTextMatch:
    """WP4 approach 1: exact name/alias/package match; ties go to the most CZI-mentioned entity."""

    def __init__(self, index):
        self.index = index

    def resolve(self, task):
        exact = [c for c in self.index.candidates(task['surface_forms']) if c['match'] == 'exact']
        if not exact:
            return {'task_id': task['task_id'], 'entity_id': None, 'score': 0.0, 'candidates': []}
        popularity = lambda c: (sum(z['mentions_count'] for z in self.index.entities[c['entity_id']]['czi']),
                                len(c['sources']))
        best = max(exact, key=lambda c: (*popularity(c), c['entity_id']))
        return {'task_id': task['task_id'], 'entity_id': best['entity_id'], 'score': 1.0,
                'candidates': [{'entity_id': c['entity_id'], 'score': popularity(c)[0]} for c in exact]}


class ContextResolver:
    def __init__(self, index, threshold=2.2, sim_weight=4.0):
        self.index, self.threshold, self.sim_weight = index, threshold, sim_weight
        docs = Counter()
        self.by_repo, self.by_package, self.by_home = {}, {}, {}
        for entity_id, entity in index.entities.items():
            docs.update(set(tokens(entity_text(entity))))
            for key in entity['repo_keys']:
                self.by_repo.setdefault(key, set()).add(entity_id)
            for ecosystem, ids in entity['package_ids'].items():
                for package in ids:
                    self.by_package.setdefault((ecosystem, package), set()).add(entity_id)
            for home in entity.get('homepages', []):
                self.by_home.setdefault(self._home(home), set()).add(entity_id)
        total = max(1, len(index.entities))
        self.idf = {t: math.log((1 + total) / (1 + n)) + 1 for t, n in docs.items()}
        self.default_idf = math.log(1 + total) + 1

    @staticmethod
    def _home(url):
        return re.sub(r'^https?://(www\.)?', '', url.strip().casefold()).rstrip('/')

    def _vector(self, words, drop):
        counts = Counter(w for w in words if w not in drop)
        vector = {w: c * self.idf.get(w, self.default_idf) for w, c in counts.items()}
        norm = math.sqrt(sum(v * v for v in vector.values())) or 1.0
        return {w: v / norm for w, v in vector.items()}

    def _url_entities(self, urls):
        found = set()
        for url in urls:
            key = repo_key(url)
            if key:
                found |= self.by_repo.get(key, set())
            package = package_ids_from_url(url)
            if package:
                found |= self.by_package.get(package, set())
            found |= self.by_home.get(self._home(url), set())
        return found

    def _match(self, entity, forms, fallback):
        names = {form_exact(n) for n in entity['names']}
        aliases = {form_exact(a) for a in entity['aliases']}
        packages = {p for ids in entity['package_ids'].values() for p in ids}
        if forms & names:
            return 'name'
        if forms & aliases:
            return 'alias'
        if forms & packages:
            return 'package'
        return fallback

    def resolve(self, task):
        forms = {form_exact(f) for f in task['surface_forms']}
        candidates = {c['entity_id']: c['match'] for c in self.index.candidates(task['surface_forms'])}
        url_hits = self._url_entities(task.get('urls', []))
        for entity_id in url_hits:
            candidates.setdefault(entity_id, 'url')
        context = ' '.join(task['contexts'])
        name_tokens = set(tokens(' '.join(task['surface_forms'])))
        context_vector = self._vector(tokens(re.sub(r'\[\[|\]\]', ' ', context)), name_tokens)
        cues = {ecosystem for ecosystem, pattern in CUES if pattern.search(context)}
        scored = []
        for entity_id, fallback in candidates.items():
            entity = self.index.entities[entity_id]
            match = self._match(entity, forms, 'czi' if fallback == 'exact' else fallback)
            entity_vector = self._vector(tokens(entity_text(entity)), name_tokens)
            similarity = sum(v * entity_vector.get(w, 0.0) for w, v in context_vector.items())
            features = {'match': match, 'similarity': round(similarity, 4), 'url': entity_id in url_hits}
            score = MATCH_WEIGHTS[match] + self.sim_weight * similarity + (3.0 if features['url'] else 0.0)
            score += 0.3 * ('wikidata' in entity['sources']) + 0.2 * (len(entity['sources']) - 1)
            # Wikipedia sitelinks separate notable software from bulk-imported namesakes.
            score += min(1.5, 0.35 * math.log1p(entity.get('sitelinks', 0)))
            ecosystems = set(entity['package_ids'])
            if cues and ecosystems:
                score += 0.6 if ecosystems & cues else -0.6
            description = entity['description'] or ''
            if WRAPPER.search(description):
                score -= 0.8
            if not description or PLACEHOLDER.search(description):
                score -= 1.0
            scored.append({'entity_id': entity_id, 'score': round(score, 4), 'features': features})
        scored.sort(key=lambda c: (-c['score'], c['entity_id']))
        best = scored[0] if scored and scored[0]['score'] >= self.threshold else None
        return {'task_id': task['task_id'], 'entity_id': best['entity_id'] if best else None,
                'score': scored[0]['score'] if scored else 0.0, 'candidates': scored[:8]}
