from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.vllm_server import resolve_vllm_base_url


class VLLMServerResolverTests(unittest.TestCase):
    def _write_server_json(self, payload: dict[str, object]) -> Path:
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        path = Path(tempdir.name) / "server.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_resolves_localhost_when_current_host_matches_fqdn(self) -> None:
        path = self._write_server_json({
            "port": 8000,
            "hostname_fqdn": "gpu-lg-cmc-h-h200-1450.host.h.pjlab.org.cn",
            "lan_ips": ["10.103.13.80"],
            "endpoints": ["http://10.103.13.80:8000/v1"],
        })

        with (
            patch("utils.vllm_server._local_hostnames", return_value={"gpu-lg-cmc-h-h200-1450"}),
            patch("utils.vllm_server._local_ips", return_value=set()),
        ):
            resolved = resolve_vllm_base_url(path)

        self.assertEqual(resolved.base_url, "http://127.0.0.1:8000/v1")
        self.assertTrue(resolved.is_local)
        self.assertEqual(resolved.source, "localhost")

    def test_resolves_lan_ip_when_current_host_does_not_match(self) -> None:
        path = self._write_server_json({
            "port": 8000,
            "hostname": "gpu-lg-cmc-h-h200-1450.host.h.pjlab.org.cn",
            "lan_ips": ["10.103.13.80", "10.103.45.21"],
            "endpoints": ["http://gpu-lg-cmc-h-h200-1450.host.h.pjlab.org.cn:8000/v1"],
        })

        with (
            patch("utils.vllm_server._local_hostnames", return_value={"another-host"}),
            patch("utils.vllm_server._local_ips", return_value={"10.0.0.2"}),
        ):
            resolved = resolve_vllm_base_url(path)

        self.assertEqual(resolved.base_url, "http://10.103.13.80:8000/v1")
        self.assertFalse(resolved.is_local)
        self.assertEqual(resolved.source, "lan_ips")

    def test_falls_back_to_non_loopback_endpoint(self) -> None:
        path = self._write_server_json({
            "endpoints": [
                "http://127.0.0.1:9000/v1",
                "http://10.103.13.80:9000/v1",
            ],
        })

        with (
            patch("utils.vllm_server._local_hostnames", return_value={"another-host"}),
            patch("utils.vllm_server._local_ips", return_value=set()),
        ):
            resolved = resolve_vllm_base_url(path, prefer_lan=False)

        self.assertEqual(resolved.base_url, "http://10.103.13.80:9000/v1")
        self.assertEqual(resolved.source, "endpoints")


if __name__ == "__main__":
    unittest.main()
