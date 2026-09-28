"""Локальный Qdrant подключается по IPv4 без системного proxy."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from vinishko.pipeline.steps.vis_searcher.catalog import connect
from vinishko.pipeline.steps.vis_searcher.configs import QdrantConfig
from vinishko.pipeline.steps.vis_searcher.snapshots import download_snapshot, upload_snapshot


class QdrantConnectionTests(unittest.TestCase):
    def test_local_connection_bypasses_ipv6_and_proxy(self):
        with patch("vinishko.pipeline.steps.vis_searcher.catalog.QdrantClient") as client:
            for host in ("localhost", "127.0.0.1", "::1", "qdrant.example.com"):
                connect(QdrantConfig(collection="test", host=host))
                kwargs = client.call_args.kwargs
                self.assertEqual(kwargs["host"], "127.0.0.1" if host == "localhost" else host)
                self.assertEqual(kwargs["trust_env"], host == "qdrant.example.com")
            connect(QdrantConfig(collection="test", path=Path("local-qdrant")))
            self.assertEqual(client.call_args.kwargs, {"path": "local-qdrant"})

    def test_snapshot_transfers_bypass_local_proxy(self):
        with (
            TemporaryDirectory() as directory,
            patch("vinishko.pipeline.steps.vis_searcher.snapshots.httpx.stream") as stream,
            patch("vinishko.pipeline.steps.vis_searcher.snapshots.httpx.post") as post,
        ):
            response = stream.return_value.__enter__.return_value
            response.is_error = post.return_value.is_error = False
            response.iter_bytes.side_effect = lambda _: iter([b"snapshot"])
            path = Path(directory) / "test.snapshot"
            snapshot = SimpleNamespace(name=path.name, size=8)
            for host in ("localhost", "127.0.0.1", "::1", "qdrant.example.com"):
                with self.subTest(host=host):
                    cfg = QdrantConfig(collection="test", host=host)
                    checksum = download_snapshot(cfg, "test", snapshot, path, 10)
                    upload_snapshot(cfg, "test", path, checksum, 10)
                    expected = host == "qdrant.example.com"
                    self.assertEqual(stream.call_args.kwargs["trust_env"], expected)
                    self.assertEqual(post.call_args.kwargs["trust_env"], expected)


if __name__ == "__main__":
    unittest.main()
