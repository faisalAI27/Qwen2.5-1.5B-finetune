"""Record declared upstream license metadata without downloading dataset rows."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import urllib.request


def main():
    root = Path(__file__).resolve().parents[1]
    sources = json.loads((root / "configs/sources.json").read_text())
    requests = [("model", "models"), ("training_dataset", "datasets"), ("test_dataset", "datasets")]
    def inspect(item):
        name, kind = item
        source = sources[name]
        url = f"https://huggingface.co/api/{kind}/{source['id']}/revision/{source['revision']}"
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                card = json.load(response).get("cardData", {}) or {}
            return {"source": name, "id": source["id"], "revision": source["revision"], "metadata_url": url,
                    "declared_license": card.get("license"), "license_name": card.get("license_name"), "license_link": card.get("license_link")}
        except Exception as error:
            return {"source": name, "id": source["id"], "metadata_url": url, "inspection_error": str(error)}
    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(inspect, requests))
    report = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(), "records": records,
              "scope": "Declared upstream metadata only; no dataset answers/solutions fetched and no public release performed.",
              "project_release_license": None,
              "release_instruction": "Preserve upstream attribution and review the recorded upstream terms plus an explicit project license before public distribution. Reacquire base/data from pinned sources; do not commit or redistribute them in this repository."}
    (root / "docs/license_metadata.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
