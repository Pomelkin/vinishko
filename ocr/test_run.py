"""Small contract tests for the Unlimited-OCR client and postprocessor."""

from __future__ import annotations

import io
import json
import os
import tempfile
import threading
import unittest
from contextlib import redirect_stderr
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from typing import cast
from unittest.mock import patch

from openai import DefaultHttpxClient
from openai import OpenAI
from PIL import Image

from ocr.run import (
    DEFAULT_RESULTS_DIR,
    RunError,
    build_request,
    clean_unlimited_ocr_output,
    default_output_path,
    derived_output_path,
    image_data_url,
    image_variants,
    iter_results,
    load_environment,
    main,
    parse_raw_results,
    recognize,
)


class _MockVllmHandler(BaseHTTPRequestHandler):
    """Capture one OpenAI request and return a minimal chat completion."""

    request_body: dict[str, Any] | None = None

    def do_POST(self) -> None:
        """Handle the chat-completions request used by the SDK test."""
        length = int(self.headers["Content-Length"])
        type(self).request_body = json.loads(self.rfile.read(length))
        response = json.dumps(
            {
                "id": "chatcmpl-mock",
                "object": "chat.completion",
                "created": 0,
                "model": "baidu/Unlimited-OCR",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "<|ref|>TEST<|/ref|>",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Keep the test output quiet."""


class UnlimitedOcrOutputTests(unittest.TestCase):
    """Verify cleanup of the grounding formats documented by Baidu and vLLM."""

    def test_unwraps_refs_and_drops_coordinate_blocks(self) -> None:
        """Reference text remains while coordinate blocks and end tokens disappear."""
        raw = (
            "<|ref|>CHATEAU TAMAGNE\n2021<|/ref|>"
            "<|det|>[[12, 34, 56, 78]]<|/det|>"
            "<｜end▁of▁sentence｜>"
        )
        self.assertEqual(clean_unlimited_ocr_output(raw), "CHATEAU TAMAGNE\n2021")

    def test_preserves_content_after_model_card_det_marker(self) -> None:
        """Text following the model-card det prefix is not discarded."""
        raw = "<|det|>text [[1, 2, 3, 4]]<|/det|>КУБАНЬ\nВИНО"
        self.assertEqual(clean_unlimited_ocr_output(raw), "КУБАНЬ\nВИНО")

    def test_coordinate_blocks_separate_adjacent_reference_text(self) -> None:
        """Grounded text boxes do not become one concatenated token."""
        raw = (
            "<|ref|>CHATEAU<|/ref|><|det|>[[1,2,3,4]]<|/det|>"
            "<|ref|>TAMAGNE<|/ref|><|det|>[[5,6,7,8]]<|/det|>"
        )
        self.assertEqual(clean_unlimited_ocr_output(raw), "CHATEAU\nTAMAGNE")

    def test_compacts_excess_blank_lines(self) -> None:
        """Postprocessing keeps paragraphs but removes excess blank lines."""
        raw = "FIRST\n\n\nSECOND\n"
        self.assertEqual(clean_unlimited_ocr_output(raw), "FIRST\n\nSECOND")


class UnlimitedOcrRequestTests(unittest.TestCase):
    """Lock down request fields and image encoding required by vLLM."""

    def test_request_matches_vllm_recipe(self) -> None:
        """The request matches the official single-image serving recipe."""
        request = build_request(
            "baidu/Unlimited-OCR", "data:image/png;base64,AA==", 8192
        )
        self.assertEqual(request["model"], "baidu/Unlimited-OCR")
        self.assertEqual(request["temperature"], 0.0)
        self.assertEqual(request["max_tokens"], 8192)
        self.assertEqual(
            request["extra_body"],
            {
                "skip_special_tokens": False,
                "vllm_xargs": {"ngram_size": 35, "window_size": 128},
            },
        )
        self.assertEqual(
            request["messages"][0]["content"][0],
            {"type": "text", "text": "<image>document parsing."},
        )
        self.assertNotIn("response_format", request)

    def test_dotenv_loads_without_overriding_process_environment(self) -> None:
        """The repository .env is loaded while explicit shell values win."""
        with tempfile.TemporaryDirectory() as directory:
            dotenv_path = Path(directory) / ".env"
            dotenv_path.write_text(
                "OCR_BASE_URL=http://from-file:8000/v1\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                self.assertTrue(load_environment(dotenv_path))
                self.assertEqual(os.environ["OCR_BASE_URL"], "http://from-file:8000/v1")
                os.environ["OCR_BASE_URL"] = "http://from-shell:8000/v1"
                dotenv_path.write_text(
                    "OCR_BASE_URL=http://changed-file:8000/v1\n",
                    encoding="utf-8",
                )
                self.assertTrue(load_environment(dotenv_path))
                self.assertEqual(
                    os.environ["OCR_BASE_URL"], "http://from-shell:8000/v1"
                )

    def test_openai_sdk_serializes_vllm_fields_at_top_level(self) -> None:
        """OpenAI extra_body reaches the vLLM endpoint in the expected shape."""
        _MockVllmHandler.request_body = None
        server = ThreadingHTTPServer(("127.0.0.1", 0), _MockVllmHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = OpenAI(
                api_key="EMPTY",
                base_url=f"http://127.0.0.1:{server.server_port}/v1",
                timeout=2,
                max_retries=0,
                http_client=DefaultHttpxClient(trust_env=False),
            )
            response = client.chat.completions.create(
                **build_request(
                    "baidu/Unlimited-OCR", "data:image/png;base64,AA==", 8192
                )
            )
            self.assertEqual(response.choices[0].message.content, "<|ref|>TEST<|/ref|>")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        body = _MockVllmHandler.request_body
        if body is None:
            self.fail("mock vLLM server did not receive a request")
        self.assertIs(body["skip_special_tokens"], False)
        self.assertEqual(body["vllm_xargs"], {"ngram_size": 35, "window_size": 128})
        self.assertNotIn("extra_body", body)

    def test_mime_comes_from_signature_not_extension(self) -> None:
        """A misleading extension does not produce the wrong data-URL MIME."""
        with tempfile.TemporaryDirectory() as directory:
            misleading_path = Path(directory) / "actually-png.jpg"
            Image.new("RGB", (2, 2), "white").save(misleading_path, format="PNG")
            self.assertTrue(
                image_data_url(misleading_path).startswith("data:image/png;base64,")
            )

    def test_center_label_preprocessing_makes_one_jpeg_crop(self) -> None:
        """The diagnostic crop is one image, not a vLLM multi-image request."""
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "portrait.png"
            Image.new("RGB", (100, 200), "white").save(image_path)
            variants = image_variants(image_path, "center-label")

        self.assertEqual([name for name, _ in variants], ["center-label"])
        self.assertTrue(variants[0][1].startswith("data:image/jpeg;base64,"))

    def test_label_scan_keeps_each_view_as_a_separate_request(self) -> None:
        """The scan exposes four single-image views for later aggregation."""
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "portrait.jpg"
            Image.new("RGB", (100, 200), "white").save(image_path)
            variants = image_variants(image_path, "label-scan")

        self.assertEqual(
            [name for name, _ in variants],
            ["original", "upper-label", "center-label", "lower-label"],
        )

    def test_default_output_is_timestamped_under_results(self) -> None:
        """Omitting --output still produces a stable, Windows-safe location."""
        output = default_output_path()
        self.assertEqual(output.parent, DEFAULT_RESULTS_DIR)
        self.assertRegex(output.name, r"^unlimited-ocr-\d{8}-\d{6}\.raw\.jsonl$")
        self.assertEqual(
            derived_output_path(output, "parsed").name,
            output.name.replace(".raw.jsonl", ".parsed.jsonl"),
        )

    def test_parallel_iterator_returns_every_query(self) -> None:
        """The concurrency branch yields one result for every selected record."""

        def fake_recognize(
            client: OpenAI,
            model: str,
            record: dict[str, Any],
            max_tokens: int,
            preprocess: str,
        ) -> dict[str, Any]:
            return {
                "query_id": record["query_id"],
                "error": None,
                "latency_ms": 1.0,
            }

        records = [{"query_id": f"ocr-{index}"} for index in range(4)]
        with (
            patch("ocr.run.recognize", side_effect=fake_recognize),
            redirect_stderr(io.StringIO()),
        ):
            results = list(
                iter_results(
                    cast(OpenAI, object()),
                    "baidu/Unlimited-OCR",
                    records,
                    8192,
                    concurrency=2,
                )
            )
        self.assertEqual(
            {result["query_id"] for result in results},
            {record["query_id"] for record in records},
        )


class RawArtifactTests(unittest.TestCase):
    """Ensure parsing is a derived operation and raw responses stay immutable."""

    def test_parser_writes_a_sibling_without_changing_raw_bytes(self) -> None:
        """Parsing preserves raw bytes and stores only cleaned text downstream."""
        with tempfile.TemporaryDirectory() as directory:
            raw_path = Path(directory) / "sample.raw.jsonl"
            parsed_path = Path(directory) / "sample.parsed.jsonl"
            raw_path.write_text(
                json.dumps(
                    {
                        "query_id": "ocr-1",
                        "raw_output": (
                            "<|ref|>CHATEAU<|/ref|><|det|>[[1,2,3,4]]<|/det|>"
                        ),
                        "error": None,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            before = raw_path.read_bytes()

            parsed = parse_raw_results(raw_path, parsed_path)

            self.assertEqual(raw_path.read_bytes(), before)
            self.assertEqual(parsed[0]["raw_text"], "CHATEAU")
            self.assertNotIn("raw_output", parsed[0])
            self.assertTrue(parsed_path.is_file())

    def test_recognizer_does_not_parse_the_raw_record(self) -> None:
        """Model collection stores raw markup and leaves cleanup to the next stage."""
        response = SimpleNamespace(
            model="baidu/Unlimited-OCR",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="<|ref|>TEST<|/ref|>"),
                    finish_reason="stop",
                )
            ],
            usage=None,
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **_kwargs: response)
            )
        )
        with patch(
            "ocr.run.image_variants",
            return_value=[("original", "data:image/png;base64,AA==")],
        ):
            result = recognize(
                cast(OpenAI, client),
                "baidu/Unlimited-OCR",
                {"query_id": "ocr-1", "image_path": "unused.jpg"},
                8192,
            )

        self.assertNotIn("raw_text", result)
        self.assertNotIn("raw_text", result["attempts"][0])
        self.assertEqual(result["raw_output"], "<|ref|>TEST<|/ref|>")

    def test_parser_rejects_raw_path_even_with_overwrite(self) -> None:
        """The overwrite switch never grants permission to replace raw JSONL."""
        with tempfile.TemporaryDirectory() as directory:
            raw_path = Path(directory) / "sample.raw.jsonl"
            raw_path.write_text('{"query_id":"ocr-1","raw_output":"TEST"}\n')
            with self.assertRaisesRegex(RunError, "must not overwrite"):
                parse_raw_results(raw_path, raw_path, overwrite=True)

    def test_unified_cli_parses_and_evaluates_existing_raw_file(self) -> None:
        """One raw-input command produces both parsed JSONL and evaluation report."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path = root / "dataset.jsonl"
            raw_path = root / "sample.raw.jsonl"
            parsed_path = root / "sample.parsed.jsonl"
            report_path = root / "sample.report.json"
            dataset_path.write_text(
                json.dumps(
                    {
                        "query_id": "ocr-1",
                        "image_path": "unused.jpg",
                        "catalog_status": "found",
                        "gold_features": [
                            {"kind": "brand", "value": "TEST", "role": "identity"}
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            raw_path.write_text(
                json.dumps(
                    {
                        "query_id": "ocr-1",
                        "raw_output": "<|ref|>TEST<|/ref|>",
                        "latency_ms": 1.0,
                        "finish_reason": "stop",
                        "error": None,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            raw_before = raw_path.read_bytes()
            argv = [
                "ocr/run.py",
                "--raw-input",
                str(raw_path),
                "--dataset",
                str(dataset_path),
                "--parsed-output",
                str(parsed_path),
                "--report",
                str(report_path),
                "--only-predicted",
            ]

            with (
                patch("sys.argv", argv),
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()),
            ):
                main()

            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(raw_path.read_bytes(), raw_before)
            self.assertEqual(report["summary"]["text_recall"], 1.0)
            self.assertTrue(parsed_path.is_file())


if __name__ == "__main__":
    unittest.main()
