"""Identifier and surface-form normalization. Names never establish identity."""

import re
import unicodedata
from urllib.parse import unquote, urlsplit

REPO_HOSTS = {'github.com', 'gitlab.com', 'bitbucket.org', 'codeberg.org', 'sourceforge.net'}
PACKAGE_PATTERNS = [
    ('pypi', re.compile(r'^(?:pypi\.org/project|pypi\.python\.org/pypi)/([^/]+)')),
    ('cran', re.compile(r'^cran\.(?:r-project\.org|rstudio\.com)/(?:package=|web/packages/)([^/]+)')),
    ('bioconductor', re.compile(r'^bioconductor\.org/packages/(?:(?:release|devel|[\d.]+)/(?:bioc|data/annotation|data/experiment|workflows)/html/)?([^/]+?)(?:\.html)?(?:/|$)')),
]


def form_exact(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text).casefold()).strip()


def form_loose(text):
    return re.sub(r'[\W_]+', '', form_exact(text)) or None


def pypi_name(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def package_id(ecosystem, name):
    return pypi_name(name) if ecosystem == 'pypi' else name.strip().lower()


def _host_path(url):
    if not isinstance(url, str) or not url.strip():
        return None, []
    url = url.strip()
    url = re.sub(r'^git\+', '', url)
    url = re.sub(r'^git@([^:]+):', r'https://\1/', url)
    if '://' not in url:
        url = 'https://' + url
    parts = urlsplit(url)
    host = (parts.hostname or '').lower().removeprefix('www.')
    return host, [unquote(p) for p in parts.path.split('/') if p]


def repo_key(url):
    host, path = _host_path(url)
    if host not in REPO_HOSTS:
        return None
    if host == 'sourceforge.net':
        if len(path) >= 2 and path[0] in ('projects', 'p'):
            return f'sourceforge.net/{path[1].lower()}'
        return None
    if host == 'gitlab.com' and '-' in path:
        path = path[:path.index('-')]
    elif host != 'gitlab.com':
        path = path[:2]
    if len(path) < 2:
        return None
    path[-1] = path[-1].removesuffix('.git')
    return '/'.join([host] + [p.lower() for p in path])


def package_ids_from_url(url):
    host, path = _host_path(url)
    if not host:
        return None
    target = '/'.join([host] + path)
    for ecosystem, pattern in PACKAGE_PATTERNS:
        match = pattern.match(target)
        if match:
            return ecosystem, package_id(ecosystem, match.group(1))
    return None
