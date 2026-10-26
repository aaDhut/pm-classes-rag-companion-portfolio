"""Fallback for flaky DNS on this Mac (for the local ingest scripts only).

Every so often macOS's resolver fails a lookup for the Qdrant host for minutes at a time
("nodename nor servname provided") while other sites resolve fine. When that happens,
ask DNS directly with the `host` command. TLS still checks the real hostname, so this
only changes how the address is found."""
import re
import socket
import subprocess

_original = socket.getaddrinfo
_cache: dict[str, list[str]] = {}


def _lookup(host: str) -> list[str]:
    if host not in _cache:
        out = subprocess.run(["host", "-t", "A", host], capture_output=True, text=True, timeout=15).stdout
        _cache[host] = re.findall(r"has address (\d+\.\d+\.\d+\.\d+)", out)
    return _cache[host]


def _getaddrinfo(host, port, *args, **kwargs):
    try:
        return _original(host, port, *args, **kwargs)
    except socket.gaierror:
        if not isinstance(host, str) or not (ips := _lookup(host)):
            raise
        return _original(ips[0], port, *args, **kwargs)


def install() -> None:
    socket.getaddrinfo = _getaddrinfo
