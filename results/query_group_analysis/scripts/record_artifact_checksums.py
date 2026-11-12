"""Record SHA-256 digests and sizes for generated query-group artifacts."""
from __future__ import annotations

import json
from pathlib import Path

try:
    from .common import OUT, RECON, sha256_file, write_json_new
except ImportError:
    from common import OUT, RECON, sha256_file, write_json_new


def main() -> None:
    """Create a non-overwriting checksum inventory after all analysis outputs exist."""
    candidates = sorted(path for path in OUT.rglob("*") if path.is_file() and path.name != "artifact_checksums.json")
    if not candidates:
        raise FileNotFoundError("No query-group analysis artifacts exist")
    entries = [{
        "path": path.relative_to(OUT).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    } for path in candidates]
    payload = {
        "checksum_manifest_version": "sha256-v1",
        "hash_algorithm": "SHA-256",
        "artifact_count": len(entries),
        "artifacts": entries,
        "note": "The reconstruction manifest records frozen input/source/model provenance; this companion inventory records generated artifact checksums without self-hashing.",
    }
    write_json_new(RECON / "artifact_checksums.json", payload)
    print(json.dumps({"artifact_count": len(entries), "checksum_manifest": str(RECON / "artifact_checksums.json")}, indent=2))


if __name__ == "__main__":
    main()
