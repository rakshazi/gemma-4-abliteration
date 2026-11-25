#!/usr/bin/env python3
"""Rebuild heretic's reproduce/ folder offline for an already-uploaded model."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

try:
    import heretic  # noqa: F401
    from heretic.config import Settings
    from heretic.modifiers.abliteration import Abliteration
    from heretic.plugin import load_plugin
    from heretic.scorer import Scorer
    from heretic.utils import create_reproduce_folder
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "heretic-src" / "src"))
    from heretic.config import Settings
    from heretic.modifiers.abliteration import Abliteration
    from heretic.plugin import load_plugin
    from heretic.scorer import Scorer
    from heretic.utils import create_reproduce_folder

import huggingface_hub
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock


def build_settings(config_path: str) -> Settings:
    os.environ["HERETIC_CONFIG_FILE"] = config_path
    argv = sys.argv
    sys.argv = argv[:1]
    try:
        return Settings()
    finally:
        sys.argv = argv


def load_settings(config_path: str, journal: Path) -> tuple[Settings, optuna.Study]:
    study = optuna.load_study(
        study_name="heretic",
        storage=JournalStorage(
            JournalFileBackend(str(journal), lock_obj=JournalFileOpenLock(str(journal)))
        ),
    )
    raw = study.user_attrs.get("settings")
    if raw is None:
        raise SystemExit(f"{journal} has no 'settings' user attribute")
    settings = Settings.model_validate_init_only(
        json.loads(raw), build_settings(config_path)
    )
    return settings, study


def dataset_specifications(
    settings: Settings, kind: str, entries: list[Any], base_class: type
) -> list[Any]:
    specifications: list[Any] = []
    for entry in entries:
        cls = load_plugin(name=entry.plugin, base_class=base_class)
        instance_name = entry.instance_name or None
        raw = cls.get_settings_raw(settings.model_extra, kind, instance_name)
        instance = cls(heretic_settings=settings, settings=cls.validate_settings(raw))
        specifications.extend(instance.get_dataset_specifications())
    return specifications


def uploaded_hashes(repo_id: str) -> dict[str, str]:
    info = huggingface_hub.HfApi().model_info(repo_id=repo_id, files_metadata=True)
    hashes = {}
    for sibling in info.siblings or []:
        if sibling.rfilename.endswith(".safetensors"):
            sha256 = getattr(sibling.lfs, "sha256", None) if sibling.lfs else None
            if not sha256:
                raise SystemExit(f"{repo_id}: no LFS sha256 for {sibling.rfilename}")
            hashes[sibling.rfilename] = sha256
    if not hashes:
        raise SystemExit(f"{repo_id}: no safetensors found, upload the weights first")
    return hashes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.ega-all-heretic.toml")
    parser.add_argument("--journal", default=None, help="defaults to <checkpoint_dir>/<model>.jsonl")
    parser.add_argument("--repo-id", required=True, help="the uploaded model repo")
    parser.add_argument("--base-model", required=True, help="Hub id of the stage-1 prebake")
    parser.add_argument(
        "--trial-number",
        type=int,
        required=True,
        help="study trial number; the journal's trial_index is a front index, so map by score",
    )
    parser.add_argument("--reproducibility", choices=["full", "basic"], default="full")
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    probe = build_settings(args.config)
    journal = Path(
        args.journal or Path(probe.study_checkpoint_dir) / f"{probe.model.replace('/', '--')}.jsonl"
    )
    if not journal.exists():
        raise SystemExit(f"journal not found: {journal}")

    settings, study = load_settings(args.config, journal)
    settings.model = args.base_model
    settings.n_trials = len(study.trials)

    trial = study.trials[args.trial_number]
    if trial.state != optuna.trial.TrialState.COMPLETE:
        raise SystemExit(f"trial {args.trial_number} is {trial.state.name}; pick a COMPLETE one")
    for candidate in study.trials:
        marker = ">" if candidate.number == trial.number else " "
        print(
            f"  {marker} trial {candidate.number}: "
            f"{candidate.values[0] * 100:.0f}/100 refusals, KL {candidate.values[1]:.5f}"
        )

    modifier_class = Abliteration.__mro__[1]
    specifications = dataset_specifications(
        settings, "scorer", settings.scorers, Scorer
    ) + dataset_specifications(settings, "modifier", settings.modifiers, modifier_class)

    print(f"journal:  {journal}")
    print(f"study:    {len(study.trials)} trials, released trial {args.trial_number}")
    print(f"base:     {settings.model} (stage-1 prebake)")
    print(f"hashes:   fetching for {args.repo_id}")
    hashes = uploaded_hashes(args.repo_id)
    print(f"datasets: {len(specifications)} pinned specifications")

    if args.dry_run:
        for specification in specifications:
            print(f"  - {specification.dataset} @ {specification.commit[:12]} {specification.split}")
        print("dry run, nothing written")
        return 0

    out_dir = Path(args.out_dir).resolve()
    shutil.rmtree(out_dir / "reproduce", ignore_errors=True)
    create_reproduce_folder(
        out_dir,
        settings,
        specifications,
        checkpoint_path=journal,
        trial=trial,
        uploaded_model_hashes=hashes,
        include_system_information=(args.reproducibility == "full"),
    )

    print(f"\nwrote {out_dir / 'reproduce'}:")
    for file in sorted((out_dir / "reproduce").iterdir()):
        print(f"  {file.name:32} {file.stat().st_size:>9,} B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
