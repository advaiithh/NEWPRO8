"""Offline tripwire: any non-loopback socket destination raises and is counted."""
from __future__ import annotations

import ipaddress
import socket
import threading
from typing import Any


class OfflineViolation(OSError):
    pass


class _State:
    attempts = 0
    last: str = ""
    installed = False
    lock = threading.Lock()


_orig: dict[str, Any] = {}


def _is_loopback(host: Any) -> bool:
    if isinstance(host, bytes):
        host = host.decode()
    if host in ("localhost", "", None):
        return True
    try:
        return ipaddress.ip_address(str(host).split("%")[0]).is_loopback
    except ValueError:
        return False


def _violate(host: Any) -> None:
    with _State.lock:
        _State.attempts += 1
        _State.last = str(host)
    raise OfflineViolation(f"offline mode: blocked network access to {host!r}")


def _dest(address: Any) -> Any:
    return address[0] if isinstance(address, tuple) else address


def install() -> None:
    if _State.installed:
        return
    _orig.update(connect=socket.socket.connect, connect_ex=socket.socket.connect_ex, gai=socket.getaddrinfo)

    def connect(self, address):  # type: ignore[no-untyped-def]
        if self.family in (socket.AF_INET, socket.AF_INET6) and not _is_loopback(_dest(address)):
            _violate(_dest(address))
        return _orig["connect"](self, address)

    def connect_ex(self, address):  # type: ignore[no-untyped-def]
        if self.family in (socket.AF_INET, socket.AF_INET6) and not _is_loopback(_dest(address)):
            _violate(_dest(address))
        return _orig["connect_ex"](self, address)

    def gai(host, *a, **k):  # type: ignore[no-untyped-def]
        if not _is_loopback(host):
            _violate(host)
        return _orig["gai"](host, *a, **k)

    socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo = connect, connect_ex, gai
    _State.installed = True


def uninstall() -> None:
    if _State.installed:
        socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo = _orig["connect"], _orig["connect_ex"], _orig["gai"]
        _State.installed = False


def status() -> dict[str, Any]:
    return {"installed": _State.installed, "attempts": _State.attempts, "last": _State.last}


def reset() -> None:
    _State.attempts, _State.last = 0, ""
