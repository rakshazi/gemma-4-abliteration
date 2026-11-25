#!/usr/bin/env python3
"""Set or unset model_action and trial_index in a finished study's journal."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone

from common import load_settings
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock

# The two fields the journal restores over the config, deciding whether menus appear.
DRIVEN_FIELDS = ("model_action", "trial_index")


def journal_path(model: str, study_checkpoint_dir: str) -> str:
    """Replicate heretic's journal filename rule (main.py)."""
    stem = "".join(c if (c.isalnum() or c in ["_", "-"]) else "--" for c in model)
    return os.path.join(study_checkpoint_dir, stem + ".jsonl")


def last_settings_from_journal(path: str) -> dict | None:
    """Read the last settings user attribute from the journal file itself."""
    last = None
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            settings = (json.loads(line).get("user_attr") or {}).get("settings")
            if settings is not None:
                last = json.loads(settings)
    return last


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Without either option both fields are unset, which brings back the trial and "
            "action menus. Stop heretic first, it locks the journal. The previous journal is "
            "kept as '<journal>.bak-<timestamp>'; trials, Pareto front and the finished flag "
            "are untouched, so the study still holds every trial afterwards."
        ),
    )
    parser.add_argument(
        "--config",
        default="config.toml",
        help="Heretic config used for the study (default: config.toml)",
    )
    parser.add_argument(
        "--journal",
        default=None,
        help="Journal file (default: derived from the config's model and study_checkpoint_dir)",
    )
    parser.add_argument(
        "--study",
        default=None,
        help="Study name inside the journal (default: the only study in the file)",
    )
    parser.add_argument(
        "--model-action",
        choices=["save", "upload", "chat", "benchmark"],
        default=None,
        help="Action to write into the journal; omit to unset it",
    )
    parser.add_argument(
        "--trial-index",
        type=int,
        default=None,
        help="Position in the sorted Pareto front to write into the journal; omit to unset it",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the changes without writing to the journal",
    )
    args = parser.parse_args()

    settings = load_settings(config=args.config)
    path = args.journal or journal_path(settings.model, settings.study_checkpoint_dir)
    if not os.path.isfile(path):
        print(f"journal not found: {path}", file=sys.stderr)
        return 1

    lock = path + ".lock"
    if os.path.exists(lock):
        print(
            f"locking file present: {lock}\n"
            "A heretic process is probably running. Stop it first.",
            file=sys.stderr,
        )
        return 1

    desired = {"model_action": args.model_action, "trial_index": args.trial_index}

    try:
        storage = JournalStorage(
            JournalFileBackend(path, lock_obj=JournalFileOpenLock(path))
        )
        studies = storage.get_all_studies()
        if not studies:
            print(f"no study in {path}", file=sys.stderr)
            return 1
        study_name = args.study or studies[0].study_name
        study_id = storage.get_study_id_from_name(study_name)

        user_attrs = storage.get_study_user_attrs(study_id)
        raw = user_attrs.get("settings")
        if raw is None:
            print(f"study {study_name!r} has no 'settings' user attribute", file=sys.stderr)
            return 1

        patched = dict(json.loads(raw))
        changes = []
        for field, value in desired.items():
            if patched.get(field) != value:
                changes.append(f"  {field}: {patched.get(field)!r} -> {value!r}")
                patched[field] = value

        n_trials = storage.get_n_trials(study_id)
        print(f"journal: {path}")
        print(
            f"study:   {study_name!r}, {n_trials} trials, "
            f"finished={user_attrs.get('finished')}"
        )

        if not changes:
            print(f"nothing to do: {', '.join(DRIVEN_FIELDS)} already hold the requested values")
            return 0

        print("changes:")
        for line in changes:
            print(line)

        if args.dry_run:
            print("dry run, journal not written")
            return 0

        backup = f"{path}.bak-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
        shutil.copy2(path, backup)
        storage.set_study_user_attr(study_id, "settings", json.dumps(patched))
    except Exception as exc:  # noqa: BLE001 - report, never half-patch a journal
        print(f"failed to patch the journal: {exc}", file=sys.stderr)
        return 1

    check = last_settings_from_journal(path)
    assert check is not None and all(check.get(field) == value for field, value in desired.items()), (
        f"journal write did not stick, last settings read back as {check}"
    )
    print(f"backup:  {backup}")
    if all(value is None for value in desired.values()):
        print("patched: the next start shows the trial menu, then the action menu.")
    else:
        print(f"patched: the next start applies {desired} without asking.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
