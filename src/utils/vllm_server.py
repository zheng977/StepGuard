from __future__ import annotations

import json
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


@dataclass(frozen=True)
class ResolvedVLLMEndpoint:
    base_url: str
    is_local: bool
    source: str
    server_json: str


def resolve_vllm_base_url(
    server_json: str | Path,
    *,
    prefer_local: bool = True,
    prefer_lan: bool = True,
) -> ResolvedVLLMEndpoint:
    """Resolve an OpenAI-compatible vLLM base URL from a server.json file.

    If the current host appears to be the serving host, return localhost so
    users do not have to update configs after a pod restart. Otherwise return
    a LAN endpoint from the JSON, falling back to the recorded endpoints list.
    """

    path = Path(server_json)
    payload = json.loads(path.read_text(encoding="utf-8"))
    port = int(payload.get("port") or _port_from_endpoints(payload.get("endpoints")) or 8000)

    if prefer_local and _is_current_host(payload):
        return ResolvedVLLMEndpoint(
            base_url=f"http://127.0.0.1:{port}/v1",
            is_local=True,
            source="localhost",
            server_json=str(path),
        )

    if prefer_lan:
        for ip in payload.get("lan_ips") or []:
            if ip:
                return ResolvedVLLMEndpoint(
                    base_url=f"http://{ip}:{port}/v1",
                    is_local=False,
                    source="lan_ips",
                    server_json=str(path),
                )

    for endpoint in payload.get("endpoints") or []:
        if endpoint and not _is_loopback_url(str(endpoint)):
            return ResolvedVLLMEndpoint(
                base_url=str(endpoint).rstrip("/"),
                is_local=False,
                source="endpoints",
                server_json=str(path),
            )

    host = payload.get("hostname_fqdn") or payload.get("hostname") or payload.get("host") or "127.0.0.1"
    return ResolvedVLLMEndpoint(
        base_url=f"http://{host}:{port}/v1",
        is_local=_is_loopback_host(str(host)),
        source="host_port",
        server_json=str(path),
    )


def _is_current_host(payload: dict[str, Any]) -> bool:
    local_names = _local_hostnames()
    local_ips = _local_ips()
    server_names = {
        str(payload.get("hostname") or "").lower(),
        str(payload.get("hostname_fqdn") or "").lower(),
    }
    for name in list(server_names):
        if "." in name:
            server_names.add(name.split(".", 1)[0])
    server_names.discard("")
    if local_names & server_names:
        return True

    for ip in payload.get("lan_ips") or []:
        if str(ip) in local_ips:
            return True
    host = str(payload.get("host") or "")
    return host in {"127.0.0.1", "0.0.0.0", "localhost"} and bool(local_names & server_names)


def _local_hostnames() -> set[str]:
    names = {socket.gethostname().lower(), socket.getfqdn().lower()}
    for name in list(names):
        if "." in name:
            names.add(name.split(".", 1)[0])
    names.discard("")
    return names


def _local_ips() -> set[str]:
    ips = {"127.0.0.1"}
    for name in _local_hostnames():
        try:
            for item in socket.getaddrinfo(name, None):
                ips.add(str(item[4][0]))
        except socket.gaierror:
            continue
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            ips.add(sock.getsockname()[0])
    except OSError:
        pass
    return ips


def _port_from_endpoints(endpoints: Any) -> int | None:
    for endpoint in endpoints or []:
        parsed = urlparse(str(endpoint))
        if parsed.port is not None:
            return int(parsed.port)
    return None


def _is_loopback_url(url: str) -> bool:
    return _is_loopback_host(urlparse(url).hostname or "")


def _is_loopback_host(host: str) -> bool:
    return host in {"127.0.0.1", "localhost", "::1"}
