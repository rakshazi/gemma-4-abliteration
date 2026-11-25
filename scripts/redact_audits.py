#!/usr/bin/env python3
"""Replace prompts and answers in an eval audit JSON with hashes, keeping every number."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

CONTENT_KEYS = {"prompt", "response", "text", "completion", "answer", "generation"}
LONG_STRING_LIMIT = 300


def redact(value, path: str, hits: list[str]) -> object:
    if isinstance(value, dict):
        return {k: redact(v, f"{path}.{k}", hits) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v, f"{path}[{i}]", hits) for i, v in enumerate(value)]
    if isinstance(value, str) and path.rsplit(".", 1)[-1] in CONTENT_KEYS:
        hits.append(path)
        return {
            "sha256": hashlib.sha256(value.encode("utf-8", "replace")).hexdigest()[:16],
            "chars": len(value),
        }
    return value


def audit_long_strings(value, path: str, found: list[str]) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            audit_long_strings(v, f"{path}.{k}", found)
    elif isinstance(value, list):
        for i, v in enumerate(value):
            audit_long_strings(v, f"{path}[{i}]", found)
    elif isinstance(value, str) and len(value) > LONG_STRING_LIMIT:
        found.append(f"{path} ({len(value)} chars)")


def redact_file(source: Path, out_dir: Path) -> Path:
    data = json.loads(source.read_text(encoding="utf-8"))
    hits: list[str] = []
    redacted = redact(data, "$", hits)

    leftovers: list[str] = []
    audit_long_strings(redacted, "$", leftovers)
    if leftovers:
        raise SystemExit(
            f"{source}: {len(leftovers)} long string(s) survived redaction, refusing to write:\n  "
            + "\n  ".join(leftovers[:10])
        )

    redacted["redacted"] = {
        "prompts": True,
        "responses": True,
        "method": "content fields replaced by sha256 prefix and character count",
        "fields_redacted": len(hits),
        "redacted_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "source": str(source),
        "note": (
            "Prompt index refers to the row order of the pinned dataset commit listed in "
            "dataset_revisions, so any flag can be matched against the public dataset "
            "without republishing the text."
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{source.stem}-redacted.json"
    target.write_text(json.dumps(redacted, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"  {source.name} -> {target} ({len(hits)} fields redacted)")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Redact prompts and answers from eval audit JSON files, keep the numbers.",
    )
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("card/evidence/evaluations"))
    args = parser.parse_args()

    for source in args.sources:
        if not source.is_file():
            raise SystemExit(f"missing input file: {source}")
    print(f"redacting {len(args.sources)} audit file(s) into {args.out_dir}/")
    for source in args.sources:
        redact_file(source, args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
