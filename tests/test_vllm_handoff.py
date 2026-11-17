import asyncio
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
import zipfile
from unittest.mock import patch

import numpy as np
from safetensors.numpy import load_file, save_file

from s1small.budget_forcing import Budget, Protocol, run_structured
from s1small.vllm_backend import AsyncBridge, VLLMDecoder
from s1small.vllm_export import TensorFile, convert_adapter, dequantize_q4, export, verify_export, write_tensor
import s1small.vllm_export as exporting


ROOT = Path(__file__).resolve().parents[1]


def adapter_fixture(directory, layers=1, rank=2):
    generator = np.random.default_rng(7)
    tensors = {}
    for layer in range(layers):
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            stem = f"model.layers.{layer}.self_attn.{projection}"
            tensors[stem + ".lora_a"] = generator.normal(size=(64, rank)).astype(np.float32)
            tensors[stem + ".lora_b"] = generator.normal(size=(rank, 4)).astype(np.float32)
    directory.mkdir(parents=True)
    save_file(tensors, str(directory / "adapters.safetensors"))
    checksum = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        checksum.update(name.encode()); checksum.update(str(tuple(value.shape)).encode()); checksum.update(value.tobytes())
    configuration = {"fine_tune_type": "lora", "num_layers": layers,
                     "lora_parameters": {"rank": rank, "scale": 2.0, "dropout": 0.05}}
    (directory / "adapter_config.json").write_text(json.dumps(configuration))
    return tensors, configuration, checksum.hexdigest()


class ConversionTests(unittest.TestCase):
    def test_q4_little_endian_codes_and_group_scales_are_reconstructed(self):
        codes = np.tile(np.arange(16, dtype=np.uint32), 8).reshape(1, 128)
        packed = np.sum(codes.reshape(1, 16, 8) << np.arange(0, 32, 4, dtype=np.uint32), axis=-1, dtype=np.uint32)
        result = dequantize_q4(packed, np.array([[0.25, 2.0]]), np.array([[-1.0, 3.0]]))
        expected = np.concatenate([codes[:, :64] * 0.25 - 1, codes[:, 64:] * 2 + 3], axis=1)
        np.testing.assert_array_equal(result, expected)

    def test_streamed_weight_is_readable_by_independent_safetensors_library(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tensor.safetensors"
            values = np.arange(21, dtype=np.float32).reshape(7, 3) / 7
            write_tensor(path, "matrix", values.shape, "F16", [values[:3], values[3:]])
            np.testing.assert_array_equal(load_file(path)["matrix"], values.astype(np.float16))
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                write_tensor(path, "matrix", values.shape, "F16", [values[:1]])

    def test_bfloat16_storage_decodes_without_mlx_or_torch(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bf16.safetensors"
            header = json.dumps({"x": {"dtype": "BF16", "shape": [3], "data_offsets": [0, 6]}}).encode()
            path.write_bytes(struct.pack("<Q", len(header)) + header + np.array([0x3F80, 0xC000, 0x3F00], dtype="<u2").tobytes())
            np.testing.assert_array_equal(TensorFile(path).float_rows("x", 0, 3), [1.0, -2.0, 0.5])

    def test_peft_transpose_preserves_trained_low_rank_function_and_fingerprint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tensors, config, fingerprint = adapter_fixture(root / "mlx")
            record = convert_adapter(root / "mlx/adapters.safetensors", root / "peft", config, fingerprint)
            restored = load_file(root / "peft/adapter_model.safetensors")
            x = np.random.default_rng(8).normal(size=(3, 64)).astype(np.float64)
            for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
                stem = "model.layers.0.self_attn." + projection
                original = ((x @ tensors[stem + ".lora_a"]) @ tensors[stem + ".lora_b"]) * 2
                converted = ((x @ restored["base_model.model." + stem + ".lora_A.weight"].T)
                             @ restored["base_model.model." + stem + ".lora_B.weight"].T) * record["alpha"] / record["rank"]
                np.testing.assert_allclose(original, converted, rtol=1e-12, atol=1e-12)
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                convert_adapter(root / "mlx/adapters.safetensors", root / "bad", config, "wrong")

    def test_resumable_export_preserves_dense_index_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "artifacts/base-mlx4/rev-g64"; base.mkdir(parents=True)
            config = {"model_type": "qwen2", "quantization": {"bits": 4, "group_size": 64, "mode": "affine"},
                      "quantization_config": {"bits": 4}, "torch_dtype": "bfloat16"}
            (base / "config.json").write_text(json.dumps(config))
            (base / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"old": "model.safetensors"}}))
            (base / "generation_config.json").write_text('{"eos_token_id": [3]}')
            (base / "tokenizer_config.json").write_text('{}')
            packed = np.full((4, 8), 0x76543210, dtype=np.uint32)
            weights = {"model.embed_tokens.weight": packed, "model.embed_tokens.scales": np.full((4, 1), 0.25, dtype=np.float32),
                       "model.embed_tokens.biases": np.full((4, 1), -1, dtype=np.float32),
                       "model.norm.weight": np.ones(64, dtype=np.float32)}
            save_file(weights, str(base / "model.safetensors"))
            _, _, fingerprint = adapter_fixture(root / "artifacts/trained")
            (root / "configs").mkdir(); (root / "manifests").mkdir()
            (root / "configs/sources.json").write_text(json.dumps({"model": {"id": "synthetic", "revision": "rev"}}))
            (root / "manifests/training_freeze.json").write_text('{"selected_run_id": "selected"}')
            selected = root / "results/training/selected"; selected.mkdir(parents=True)
            (selected / "summary.json").write_text(json.dumps({"adapter_directory": "artifacts/trained", "adapter_tensor_sha256": fingerprint}))
            source = root / "src/s1small/vllm_export.py"; source.parent.mkdir(parents=True); source.write_text('synthetic conversion code identity')
            destination = root / "artifacts/vllm-export"
            with patch.object(exporting, "__file__", str(source)):
                first = export(root, destination)
                self.assertEqual(first, export(root, destination))
            index = json.loads((destination / "base/model.safetensors.index.json").read_text())
            self.assertNotIn("old", index["weight_map"])
            filename = index["weight_map"]["model.embed_tokens.weight"]
            expected = dequantize_q4(packed, weights["model.embed_tokens.scales"], weights["model.embed_tokens.biases"]).astype(np.float16)
            np.testing.assert_array_equal(load_file(destination / "base" / filename)["model.embed_tokens.weight"], expected)
            dense_config = json.loads((destination / "base/config.json").read_text())
            self.assertNotIn("quantization_config", dense_config)
            self.assertEqual(dense_config["torch_dtype"], "float16")
            (destination / "base" / filename).write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "Exported"):
                verify_export(root, destination)


class FakeEngine:
    def __init__(self, outputs, delay=0):
        self.outputs = iter(outputs); self.delay = delay; self.aborted = []; self.requests = []
    async def generate(self, **arguments):
        self.requests.append(arguments)
        values = next(self.outputs)
        if self.delay:
            await asyncio.sleep(self.delay)
        for index in range(1, len(values) + 1):
            yield SimpleNamespace(outputs=[SimpleNamespace(token_ids=values[:index])])
    async def abort(self, identifier):
        self.aborted.append(identifier)


class BridgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_multiple_state_machines_feed_one_engine_loop_concurrently(self):
        class ConcurrentEngine:
            def __init__(self):
                self.active = self.maximum = 0
                self.threads = set()
            async def generate(self, **arguments):
                self.active += 1
                self.maximum = max(self.maximum, self.active)
                self.threads.add(threading.get_ident())
                try:
                    await asyncio.sleep(0.02)
                    yield SimpleNamespace(outputs=[SimpleNamespace(token_ids=[98])])
                finally:
                    self.active -= 1
        engine = ConcurrentEngine()
        bridge = AsyncBridge(engine, asyncio.get_running_loop(), {"max_model_len": 100, "seed": 42}, SimpleNamespace)
        results = await asyncio.gather(*(asyncio.to_thread(bridge.generate, [i + 1], 1, {98}, time.perf_counter() + 2, None) for i in range(8)))
        self.assertEqual(engine.maximum, 8)
        self.assertEqual(engine.threads, {threading.get_ident()})
        self.assertTrue(all(r.tokens == (98,) for r in results))
        bridge.close()

    async def test_real_protocol_handles_counted_stop_wait_and_answer_through_async_bridge(self):
        loop = asyncio.get_running_loop()
        engine = FakeEngine([[10, 99], [11, 98], [20, 98]])
        bridge = AsyncBridge(engine, loop, {"max_model_len": 100, "seed": 42}, SimpleNamespace)
        tokenizer = SimpleNamespace(decode=lambda tokens, **kw: " ".join(map(str, tokens)), __len__=lambda: 200)
        class Tokenizer:
            def __len__(self): return 200
            def decode(self, tokens, **kw): return " ".join(map(str, tokens))
        decoder = VLLMDecoder(bridge, Tokenizer(), [98])
        protocol = Protocol(99, (98,), (50,), (51,), (52,))
        result = await asyncio.to_thread(run_structured, decoder, [1], Budget("E1", 10, 5, 30, 1), protocol, 2)
        self.assertEqual(result["forced_wait_count"], 1)
        self.assertEqual(result["total_generated_tokens"], 6)
        self.assertEqual(result["answer_token_ids"], [20, 98])
        self.assertEqual(result["answer_text"], "20")
        self.assertTrue(engine.requests[0]["sampling_params"].ignore_eos)
        self.assertEqual(engine.requests[0]["sampling_params"].stop_token_ids, [98, 99])
        bridge.close()

    async def test_deadline_aborts_request_and_preserves_partial_tokens(self):
        engine = FakeEngine([[10, 11]], delay=1)
        bridge = AsyncBridge(engine, asyncio.get_running_loop(), {"max_model_len": 100, "seed": 42}, SimpleNamespace)
        result = await bridge.collect([1], 10, {98}, time.perf_counter() + 0.02, None)
        self.assertTrue(result.timed_out)
        self.assertTrue(engine.aborted)
        bridge.close()

    async def test_context_is_not_truncated_and_tokens_after_stop_are_rejected(self):
        engine = FakeEngine([[10, 98, 11]])
        bridge = AsyncBridge(engine, asyncio.get_running_loop(), {"max_model_len": 100, "seed": 42}, SimpleNamespace)
        with self.assertRaisesRegex(ValueError, "no-truncation"):
            await bridge.collect([1] * 99, 10, {98}, time.perf_counter() + 1, None)
        with self.assertRaisesRegex(ValueError, "stop-token"):
            await bridge.collect([1], 10, {98}, time.perf_counter() + 1, None)
        bridge.close()
        with self.assertRaisesRegex(RuntimeError, "shutting down"):
            await asyncio.to_thread(bridge.generate, [1], 10, {98}, time.perf_counter() + 1, None)


class RunnerTests(unittest.TestCase):
    def test_vllm_notebook_has_drive_download_and_exports_results_without_weights(self):
        import ast
        notebook = json.loads((ROOT / "notebooks/runpod_inference.ipynb").read_text())
        source = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
        self.assertIn("1e0G7nG1B8c6G4CKzw6-U69my-x7GJfuD", source)
        self.assertIn("scripts/runpod_vllm.py", source)
        exporter = None
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertEqual(cell["outputs"], [])
                tree = ast.parse("".join(cell["source"]))
                for node in tree.body:
                    if isinstance(node, ast.FunctionDef) and node.name == "export_results":
                        exporter = node
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "workspace"; root.mkdir()
            for name in ("results/answer.json", "manifests/evaluation_freeze.json", "artifacts/vllm-export/export_manifest.json",
                         "artifacts/vllm-export/base/model.safetensors", ".venv/private", "results/.run.lock"):
                path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('test')
            destination = Path(temporary) / "results.zip"
            namespace = {"ROOT": root, "RESULTS_ZIP": destination, "zipfile": zipfile, "NEW_FILES": ()}
            exec(compile(ast.Module(body=[exporter], type_ignores=[]), "notebook-export", "exec"), namespace)
            namespace["export_results"]()
            with zipfile.ZipFile(destination) as archive:
                names = archive.namelist()
                self.assertIn("results/answer.json", names)
                self.assertIn("artifacts/vllm-export/export_manifest.json", names)
                self.assertFalse(any(name.endswith('.safetensors') or '.venv' in name or name.endswith('.lock') for name in names))

    def test_runner_refuses_active_mac_workspace(self):
        spec = importlib.util.spec_from_file_location("vllm_runner_test", ROOT / "scripts/runpod_vllm.py")
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary, patch.object(module, "__file__", str(Path(temporary) / "scripts/runpod_vllm.py")), patch("sys.argv", ["runpod_vllm.py", "verify"]):
            with self.assertRaisesRegex(ValueError, "never the active Mac"):
                module.main()
