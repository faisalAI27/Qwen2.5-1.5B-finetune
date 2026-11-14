"""Bounded, resumable HTTP range transfer with final upstream verification."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

from s1small.data import file_sha256


def download_ranges(url, target, size, expected_sha256, workers=8, chunk_size=32 * 1024**2):
    if not 1 <= workers <= 8 or chunk_size <= 0 or size <= 0:
        raise ValueError("Use 1-8 workers and a positive chunk size")
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if file_sha256(target) != expected_sha256:
            raise ValueError("Existing checkpoint has an incorrect upstream checksum")
        return
    part = target.with_suffix(target.suffix + ".ranges.part")
    state_path = target.with_suffix(target.suffix + ".ranges.json")
    identity = {"size": size, "sha256": expected_sha256, "chunk_size": chunk_size}
    done = {}
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["identity"] != identity or not part.exists():
            raise ValueError("Resume state differs from pinned checkpoint")
        with part.open("rb") as stream:
            for index, checksum in state["done"].items():
                start = int(index) * chunk_size
                stream.seek(start)
                payload = stream.read(min(chunk_size, size - start))
                if hashlib.sha256(payload).hexdigest() == checksum:
                    done[index] = checksum
    else:
        prefix_size = 0
        # Retain the already-downloaded contiguous prefix from HF's HTTP path.
        candidates = list((target.parent / ".cache/huggingface/download").glob(f"*{expected_sha256}*.incomplete"))
        if not part.exists() and len(candidates) == 1 and candidates[0].stat().st_size <= size:
            prefix_size = candidates[0].stat().st_size
            os.replace(candidates[0], part)
        with part.open("r+b" if part.exists() else "w+b") as stream:
            stream.truncate(size)
            stream.seek(0)
            for index in range(prefix_size // chunk_size):
                done[str(index)] = hashlib.sha256(stream.read(chunk_size)).hexdigest()

    def save_state():
        temporary = state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"identity": identity, "done": done}, sort_keys=True) + "\n")
        os.replace(temporary, state_path)

    save_state()
    count = (size + chunk_size - 1) // chunk_size

    def fetch(index):
        start, end = index * chunk_size, min((index + 1) * chunk_size, size) - 1
        request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"})
        for attempt in range(3):
            try:
                checksum = hashlib.sha256()
                received = 0
                with urllib.request.urlopen(request, timeout=60) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                        raise ValueError("Server did not return the exact requested range")
                    with part.open("r+b", buffering=0) as stream:
                        stream.seek(start)
                        while True:
                            payload = response.read(min(1024**2, end - start + 1 - received))
                            if not payload:
                                break
                            stream.write(payload)
                            checksum.update(payload)
                            received += len(payload)
                if received != end - start + 1:
                    raise ValueError("Incomplete HTTP range")
                return index, checksum.hexdigest()
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)

    print(f"Range transfer: {len(done)}/{count} verified chunks already present; {workers} workers", flush=True)
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, index) for index in range(count) if str(index) not in done]
        for future in as_completed(futures):
            index, checksum = future.result()
            done[str(index)] = checksum
            save_state()
            completed_bytes = sum(min(chunk_size, size - int(index) * chunk_size) for index in done)
            print(f"Checkpoint: {completed_bytes:,}/{size:,} bytes ({len(done)}/{count} chunks), {time.monotonic() - started:.1f}s", flush=True)
    if file_sha256(part) != expected_sha256:
        done.clear()
        save_state()
        raise ValueError("Completed checkpoint does not match the upstream SHA-256")
    os.replace(part, target)
    print("Upstream checkpoint SHA-256 verified", flush=True)
