import hashlib
import io
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

from s1small.download import download_ranges


class Response(io.BytesIO):
    def __init__(self, payload, start, end, total, status=206):
        super().__init__(payload)
        self.status = status
        self.headers = {"Content-Range": f"bytes {start}-{end}/{total}"}


class DownloadTests(unittest.TestCase):
    def test_ranges_reconstruct_and_existing_file_is_verified(self):
        content = b"0123456789abcdefghijklmnopqrstuvwxyz"
        checksum = hashlib.sha256(content).hexdigest()

        def request_response(request, timeout):
            start, end = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", request.get_header("Range")).groups())
            return Response(content[start:end + 1], start, end, len(content))

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "model.safetensors"
            with patch("s1small.download.urllib.request.urlopen", side_effect=request_response), patch("builtins.print"):
                download_ranges("https://example.invalid/model", target, len(content), checksum, workers=2, chunk_size=7)
            self.assertEqual(target.read_bytes(), content)
            with patch("s1small.download.urllib.request.urlopen", side_effect=AssertionError("No request expected")):
                download_ranges("https://example.invalid/model", target, len(content), checksum)

    def test_final_upstream_checksum_failure_never_publishes_model(self):
        content = b"wrong data"
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "model.safetensors"
            with patch("s1small.download.urllib.request.urlopen", return_value=Response(content, 0, len(content)-1, len(content))), patch("builtins.print"):
                with self.assertRaisesRegex(ValueError, "upstream SHA-256"):
                    download_ranges("https://example.invalid/model", target, len(content), "0" * 64, workers=1, chunk_size=100)
            self.assertFalse(target.exists())
            self.assertEqual(json.loads(target.with_suffix(".safetensors.ranges.json").read_text())["done"], {})

    def test_non_range_response_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "model.safetensors"
            with patch("s1small.download.urllib.request.urlopen", side_effect=lambda *args, **kwargs: Response(b"a", 0, 0, 1, status=200)), patch("s1small.download.time.sleep"), patch("builtins.print"):
                with self.assertRaisesRegex(ValueError, "exact requested range"):
                    download_ranges("https://example.invalid/model", target, 1, hashlib.sha256(b"a").hexdigest(), workers=1)
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
