from __future__ import annotations

import os
import tempfile
import threading
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from scripts.eval.run_batch_eval import (
    APIModelConfig,
    BatchEvalConfig,
    RunningVLLM,
    VLLMModelConfig,
    _build_auth_headers,
    build_eval_command,
    build_vllm_launch_command,
    detect_vllm_launch_mode,
    load_batch_eval_config,
    run_batch_eval,
    validate_batch_eval_config,
    wait_for_http_ready,
)


class BatchEvalScriptTests(unittest.TestCase):
    def test_build_auth_headers_supports_bearer_token(self) -> None:
        self.assertEqual(_build_auth_headers("EMPTY"), {"Authorization": "Bearer EMPTY"})
        self.assertEqual(_build_auth_headers(""), {})

    def test_load_batch_eval_config_defaults_output_root_and_expands_env(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "batch.yaml"
            model_dir = Path(temp_dir) / "model"
            model_dir.mkdir()
            os.environ["BATCH_API_KEY"] = "secret-key"
            config_path.write_text(
                "\n".join(
                    [
                        "benchmark: ts_bench",
                        "input: benchmarks/ts_bench",
                        "response_parser: safe_unsafe",
                        "concurrency: 8",
                        "api_models:",
                        "  - name: api-one",
                        "    model: gpt-5-mini",
                        "    base_url: https://api.example.com/v1",
                        "    api_key: ${BATCH_API_KEY}",
                        "    prompt_name: default_predict",
                        "vllm_models:",
                        "  - name: local-one",
                        "    model: local-guard",
                        f"    model_path: {model_dir}",
                        "    prompt_name: agentdog_definition",
                        "    response_parser: strict",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_batch_eval_config(config_path)

            self.assertEqual(config.output_root, str((Path.cwd() / "results" / "ts_bench").resolve()))
            self.assertEqual(config.input_path, str((Path.cwd() / "benchmarks" / "ts_bench").resolve()))
            self.assertEqual(config.response_parser, "safe_unsafe")
            self.assertEqual(config.api_models[0].api_key, "secret-key")
            self.assertEqual(config.api_models[0].prompt_name, "default_predict")
            self.assertEqual(config.vllm_models[0].model_path, str(model_dir))
            self.assertEqual(config.vllm_models[0].prompt_name, "agentdog_definition")
            self.assertEqual(config.vllm_models[0].response_parser, "strict")

    def test_load_batch_eval_config_supports_model_level_prompt_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "batch.yaml"
            model_dir = Path(temp_dir) / "model"
            model_dir.mkdir()
            prompts_dir = Path(temp_dir) / "prompts"
            prompts_dir.mkdir()
            api_prompt = prompts_dir / "api_prompt.txt"
            vllm_prompt = prompts_dir / "vllm_prompt.txt"
            global_prompt = prompts_dir / "global_prompt.txt"
            api_prompt.write_text("api prompt", encoding="utf-8")
            vllm_prompt.write_text("vllm prompt", encoding="utf-8")
            global_prompt.write_text("global prompt", encoding="utf-8")
            config_path.write_text(
                "\n".join(
                    [
                        "benchmark: ts_bench",
                        "prompt_file: prompts/global_prompt.txt",
                        "api_models:",
                        "  - name: api-one",
                        "    model: gpt-5-mini",
                        "    base_url: https://api.example.com/v1",
                        "    api_key: key",
                        "    prompt_file: prompts/api_prompt.txt",
                        "vllm_models:",
                        "  - name: local-one",
                        "    model: local-guard",
                        f"    model_path: {model_dir}",
                        "    prompt_file: prompts/vllm_prompt.txt",
                    ]
                ),
                encoding="utf-8",
            )

            with patch("scripts.eval.run_batch_eval.REPO_ROOT", Path(temp_dir)):
                config = load_batch_eval_config(config_path)

            self.assertEqual(config.prompt_file, str(global_prompt.resolve()))
            self.assertEqual(config.api_models[0].prompt_file, str(api_prompt.resolve()))
            self.assertEqual(config.vllm_models[0].prompt_file, str(vllm_prompt.resolve()))

    def test_validate_batch_eval_config_rejects_duplicate_names(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "model"
            model_dir.mkdir()
            config = BatchEvalConfig(
                benchmark="ts_bench",
                api_models=[APIModelConfig(name="dup", model="m1", base_url="http://a", api_key="k")],
                vllm_models=[VLLMModelConfig(name="dup", model="m2", model_path=str(model_dir))],
            )

            with self.assertRaisesRegex(ValueError, 'Duplicate model name "dup"'):
                validate_batch_eval_config(config)

    def test_build_eval_command_for_api_model(self) -> None:
        config = BatchEvalConfig(
            benchmark="ts_bench",
            input_path="/repo/benchmarks/ts_bench",
            output_root="/repo/results/ts_bench",
            prompt_name="default_predict",
            response_parser="safe_unsafe",
            limit=20,
            concurrency=16,
            ts_subset="workspace",
        )

        command = build_eval_command(
            config,
            backend="api",
            model="gpt-5-mini",
            base_url="https://api.example.com/v1",
            api_key="secret",
        )

        self.assertIn("--benchmark", command)
        self.assertIn("ts_bench", command)
        self.assertIn("--prompt-name", command)
        self.assertIn("default_predict", command)
        self.assertIn("--response-parser", command)
        self.assertIn("safe_unsafe", command)
        self.assertIn("--limit", command)
        self.assertIn("20", command)
        self.assertIn("--ts-subset", command)
        self.assertIn("workspace", command)

    def test_build_eval_command_prefers_model_level_response_parser(self) -> None:
        config = BatchEvalConfig(
            benchmark="at_bench",
            output_root="/repo/results/at_bench",
            response_parser="strict",
        )

        command = build_eval_command(
            config,
            backend="api",
            model="demo-model",
            base_url="https://api.example.com/v1",
            api_key="secret",
            response_parser="safe_unsafe",
        )

        parser_index = command.index("--response-parser")
        self.assertEqual(command[parser_index + 1], "safe_unsafe")

    def test_build_eval_command_prefers_model_level_prompt_name(self) -> None:
        config = BatchEvalConfig(
            benchmark="at_bench",
            output_root="/repo/results/at_bench",
            prompt_name="default",
        )

        command = build_eval_command(
            config,
            backend="api",
            model="demo-model",
            base_url="https://api.example.com/v1",
            api_key="secret",
            prompt_name="agentdog_definition",
        )

        prompt_index = command.index("--prompt-name")
        self.assertEqual(command[prompt_index + 1], "agentdog_definition")
        self.assertNotIn("--prompt-file", command)

    def test_build_eval_command_prefers_prompt_name_override_over_global_prompt_file(self) -> None:
        config = BatchEvalConfig(
            benchmark="at_bench",
            output_root="/repo/results/at_bench",
            prompt_name="default",
            prompt_file="/repo/prompts/global.txt",
        )

        command = build_eval_command(
            config,
            backend="api",
            model="demo-model",
            base_url="https://api.example.com/v1",
            api_key="secret",
            prompt_name="agentdog_definition",
        )

        prompt_index = command.index("--prompt-name")
        self.assertEqual(command[prompt_index + 1], "agentdog_definition")
        self.assertNotIn("--prompt-file", command)

    def test_build_eval_command_prefers_model_level_prompt_file(self) -> None:
        config = BatchEvalConfig(
            benchmark="at_bench",
            output_root="/repo/results/at_bench",
            prompt_name="default",
            prompt_file="/repo/prompts/global.txt",
        )

        command = build_eval_command(
            config,
            backend="api",
            model="demo-model",
            base_url="https://api.example.com/v1",
            api_key="secret",
            prompt_name="agentdog_definition",
            prompt_file="/repo/prompts/model.txt",
        )

        prompt_file_index = command.index("--prompt-file")
        self.assertEqual(command[prompt_file_index + 1], "/repo/prompts/model.txt")
        self.assertNotIn("--prompt-name", command)

    def test_build_vllm_launch_command(self) -> None:
        model = VLLMModelConfig(
            name="qwen",
            model="Qwen3Guard-Gen-8B",
            model_path="/models/qwen",
            tensor_parallel_size=4,
            gpu_memory_utilization=0.85,
            port=19000,
            max_model_len=4096,
            dtype="bfloat16",
            vllm_extra_args=["--trust-remote-code"],
        )

        with (
            patch("scripts.eval.run_batch_eval.detect_vllm_launch_mode", return_value="module"),
        ):
            command = build_vllm_launch_command(model)

        self.assertEqual(command[:4], [command[0], "-m", "vllm", "serve"])
        self.assertIn("/models/qwen", command)
        self.assertIn("--served-model-name", command)
        self.assertIn("Qwen3Guard-Gen-8B", command)
        self.assertIn("--port", command)
        self.assertIn("19000", command)
        self.assertIn("--trust-remote-code", command)

    def test_build_vllm_launch_command_supports_legacy_api_server(self) -> None:
        model = VLLMModelConfig(
            name="qwen",
            model="Qwen3Guard-Gen-8B",
            model_path="/models/qwen",
        )

        with patch("scripts.eval.run_batch_eval.detect_vllm_launch_mode", return_value="legacy_api_server"):
            command = build_vllm_launch_command(model)

        self.assertEqual(command[:3], [command[0], "-m", "vllm.entrypoints.openai.api_server"])
        self.assertIn("--model", command)
        self.assertIn("/models/qwen", command)

    def test_detect_vllm_launch_mode_prefers_legacy_api_server_when_module_main_missing(self) -> None:
        with (
            patch("scripts.eval.run_batch_eval.shutil.which", return_value=None),
            patch("scripts.eval.run_batch_eval.importlib.util.find_spec") as find_spec,
        ):
            def _fake_find_spec(name: str):  # noqa: ANN001
                if name == "vllm":
                    class _Spec:
                        origin = "/tmp/vllm/__init__.py"

                    return _Spec()
                if name == "vllm.entrypoints.openai.api_server":
                    return object()
                return None

            find_spec.side_effect = _fake_find_spec
            with patch("scripts.eval.run_batch_eval.Path.is_file", return_value=False):
                mode = detect_vllm_launch_mode()

        self.assertEqual(mode, "legacy_api_server")

    def test_run_batch_eval_aggregates_api_and_vllm_results(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "model"
            model_dir.mkdir()
            config = BatchEvalConfig(
                benchmark="at_bench",
                input_path=str((Path.cwd() / "benchmarks" / "at_bench" / "test.json").resolve()),
                output_root=str((Path.cwd() / "results" / "at_bench").resolve()),
                concurrency=4,
                api_models=[
                    APIModelConfig(
                        name="api-one",
                        model="gpt-5-mini",
                        base_url="https://api.example.com/v1",
                        api_key="secret",
                        prompt_name="default_predict",
                    )
                ],
                vllm_models=[
                    VLLMModelConfig(
                        name="local-one",
                        model="Qwen3Guard-Gen-8B",
                        model_path=str(model_dir),
                        prompt_name="agentdog_definition",
                        port=18001,
                    )
                ],
            )

            class _FakeProcess:
                returncode = None

                def poll(self) -> None:
                    return None

            fake_running = RunningVLLM(
                process=_FakeProcess(),  # type: ignore[arg-type]
                log_path=Path("/tmp/vllm.log"),
                log_handle=StringIO(),
                stream_thread=threading.Thread(target=lambda: None),
            )

            class _Completed:
                def __init__(self, returncode: int) -> None:
                    self.returncode = returncode

            with (
                patch("scripts.eval.run_batch_eval.run_subprocess", side_effect=[_Completed(0), _Completed(1)]),
                patch("scripts.eval.run_batch_eval.start_vllm_process", return_value=fake_running),
                patch("scripts.eval.run_batch_eval.wait_for_http_ready"),
                patch("scripts.eval.run_batch_eval.stop_vllm_process"),
            ):
                results = run_batch_eval(config)

            self.assertEqual(len(results), 2)
            self.assertEqual(results[0].name, "api-one")
            self.assertEqual(results[0].status, "success")
            self.assertEqual(results[1].name, "local-one")
            self.assertEqual(results[1].status, "failed")
            self.assertIn("vllm_log=/tmp/vllm.log", results[1].error or "")

    def test_wait_for_http_ready_rejects_unexpected_model_name(self) -> None:
        with patch("scripts.eval.run_batch_eval._extract_served_model_ids", return_value=["other-model"]) as mocked_extract:
            with self.assertRaisesRegex(RuntimeError, 'expected model "target-model"'):
                wait_for_http_ready(
                    "http://127.0.0.1:18000/v1",
                    timeout_seconds=1,
                    expected_model="target-model",
                    api_key="EMPTY",
                )
        mocked_extract.assert_called_with("http://127.0.0.1:18000/v1", api_key="EMPTY")


if __name__ == "__main__":
    unittest.main()
