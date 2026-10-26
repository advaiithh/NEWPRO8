"""R2: the tripwire blocks and counts every non-loopback destination; loopback stays usable."""
import socket
import threading

import pytest

from pv.guard import netguard
from pv.guard.netguard import OfflineViolation


@pytest.fixture(autouse=True)
def guard():
    netguard.install()
    netguard.reset()
    yield
    netguard.reset()


def test_external_connect_raises_and_counts():
    s = socket.socket()
    with pytest.raises(OfflineViolation):
        s.connect(("8.8.8.8", 53))
    with pytest.raises(OfflineViolation):
        s.connect_ex(("93.184.216.34", 80))
    s.close()
    assert netguard.status()["attempts"] == 2
    assert netguard.status()["last"] == "93.184.216.34"


def test_dns_lookup_blocked():
    with pytest.raises(OfflineViolation):
        socket.getaddrinfo("example.com", 80)
    assert netguard.status()["attempts"] == 1


def test_http_client_blocked():
    import httpx
    with pytest.raises(Exception):
        httpx.get("http://example.com", timeout=2)
    assert netguard.status()["attempts"] >= 1


def test_loopback_allowed():
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    t = threading.Thread(target=lambda: srv.accept()[0].close(), daemon=True)
    t.start()
    c = socket.socket()
    c.connect(("127.0.0.1", port))
    c.close()
    t.join(2)
    srv.close()
    assert netguard.status()["attempts"] == 0
    assert socket.getaddrinfo("localhost", 80)
