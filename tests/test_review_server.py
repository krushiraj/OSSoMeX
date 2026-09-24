import pytest

from research.annotations import review_server as server


def test_loopback_only_origin_host_and_csrf():
    for host in ('0.0.0.0','example.org','192.168.1.10'):
        with pytest.raises(ValueError):server.check_bind(host)
    server.check_bind('127.0.0.1')
    good={'Host':'127.0.0.1:8765','Origin':'http://127.0.0.1:8765','X-CSRF-Token':'secret'}
    assert server.authorized_request(good,8765,'secret',write=True)
    for patch in [{'Host':'evil.test:8765'},{'Origin':'https://evil.test'},{'X-CSRF-Token':'wrong'}, {'Origin':None}]:
        assert not server.authorized_request({**good,**patch},8765,'secret',write=True)
