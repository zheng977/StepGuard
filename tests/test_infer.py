from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path


TESTS_DIR = Path(__file__).resolve().parent
SRC_ROOT = TESTS_DIR.parent / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from infer.base import InferResponse
from infer.factory import InferFactory


class InferSmokeTests(unittest.TestCase):
    def test_openai_env_chat_case(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        model = os.getenv("OPENAI_MODEL")
        print("[debug] model:", model)
        print("[debug] api_key_set:", bool(api_key))
        print("[debug] base_url:", base_url)
        if not api_key or not model:
            self.skipTest(
                "Missing OPENAI_API_KEY or OPENAI_MODEL. "
                "Export OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL before running."
            )

        backend = InferFactory.create(
            "api",
            model=model,
            api_key=api_key,
            base_url=base_url,
        )
        response = backend.chat(
            [
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": "who are you?"},
            ],
            temperature=0.0,
            max_tokens=320,
            timeout=60,
        )

        self.assertIsInstance(response, InferResponse)
        self.assertEqual(response.model, model)
        print("[debug] infer_text:", response.text)
        print("[debug] finish_reason:", response.finish_reason)
        print("[debug] raw_keys:", sorted(response.raw.keys()) if isinstance(response.raw, dict) else type(response.raw))
        self.assertIsInstance(response.raw, dict)
        self.assertIn("choices", response.raw)
        self.assertTrue(len(response.raw.get("choices", [])) > 0)
        self.assertIsNotNone(response.finish_reason)


if __name__ == "__main__":
    unittest.main()
