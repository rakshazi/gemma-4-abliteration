#!/usr/bin/env python3
"""Evaluate abliterated models: refusal rates across datasets plus KL vs a baseline."""

from __future__ import annotations

import argparse
import gc
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import torch
import torch.nn.functional as F
from rich.markup import escape
from rich.table import Table

from common import (
    environment_report,
    get_keyword_markers,
    get_scorer_prompt_specs,
    load_settings,
    match_refusal,
    write_json,
)
from heretic.model import Model
from heretic.system import empty_cache
from heretic.utils import format_duration, load_prompts, print


@dataclass
class EvalDataset:
    name: str
    prompts: list[str]
    categories: list[str] | None = None
    systems: list[str] | None = field(default=None)


def load_eval_dataset(
    name: str,
    limit: int = 0,
    revision: str | None = None,
) -> EvalDataset:
    from datasets import load_dataset

    if name == "mlabonne":
        dataset = load_dataset(
            "mlabonne/harmful_behaviors", split="test", revision=revision
        )
        prompts = list(dataset["text"])
        categories = None
    elif name == "jbb":
        dataset = load_dataset(
            "JailbreakBench/JBB-Behaviors",
            "behaviors",
            split="harmful",
            revision=revision,
        )
        prompts = list(dataset["Goal"])
        categories = list(dataset["Category"])
    elif name == "tulu":
        dataset = load_dataset(
            "allenai/tulu-3-harmbench-eval", split="test", revision=revision
        )
        prompts = list(dataset["Behavior"])
        categories = list(dataset["SemanticCategory"])
    elif name == "nous":
        dataset = load_dataset(
            "NousResearch/RefusalDataset", split="train", revision=revision
        )
        prompts = list(dataset["prompt"])
        categories = list(dataset["overarching_category"])
    else:
        raise ValueError(f"unknown dataset: {name}")

    if limit:
        prompts = prompts[:limit]
        if categories is not None:
            categories = categories[:limit]

    return EvalDataset(name=name, prompts=prompts, categories=categories)


def load_text_prompts(path: str, limit: int = 0) -> EvalDataset:
    with open(path, encoding="utf-8") as handle:
        prompts = [line.strip() for line in handle if line.strip()]
    if limit:
        prompts = prompts[:limit]
    return EvalDataset(name=os.path.basename(path), prompts=prompts)


def load_model(model_path: str, args: argparse.Namespace) -> Model:
    values: dict[str, Any] = {"device_map": args.device_map}
    if args.dtype:
        values["dtypes"] = args.dtype
    if args.quantization:
        values["quantization"] = args.quantization
    if args.batch_size is not None:
        values["batch_size"] = args.batch_size
    if args.response_prefix is not None:
        values["response_prefix"] = args.response_prefix
    if args.max_new_tokens is not None:
        values["max_response_length"] = args.max_new_tokens
    if args.system_prompt is not None:
        values["system_prompt"] = args.system_prompt
    values = {key: value for key, value in values.items() if value is not None}

    settings = load_settings(model=model_path, config=args.config, values=values)
    if settings.batch_size == 0:
        settings.batch_size = 4
    return Model(settings)


def evaluate_refusals(
    model: Model,
    dataset: EvalDataset,
    markers: list[str],
) -> dict[str, Any]:
    from heretic.utils import Prompt

    systems = dataset.systems or [model.settings.system_prompt] * len(dataset.prompts)
    prompts = [
        Prompt(system=system, user=prompt)
        for system, prompt in zip(systems, dataset.prompts)
    ]
    started = time.perf_counter()
    responses = model.get_responses_batched(prompts, skip_special_tokens=True)
    elapsed = time.perf_counter() - started

    flagged: list[dict[str, Any]] = []
    per_prompt: list[bool] = []
    for index, (prompt, response) in enumerate(zip(dataset.prompts, responses)):
        marker = match_refusal(response, markers)
        per_prompt.append(marker is not None)
        if marker is not None:
            flagged.append(
                {
                    "index": index,
                    "marker": marker,
                    "prompt": prompt,
                    "response": " ".join(response.split())[:1200],
                    "category": dataset.categories[index]
                    if dataset.categories
                    else None,
                }
            )

    categories: dict[str, dict[str, float]] = {}
    if dataset.categories is not None:
        grouped: dict[str, list[bool]] = {}
        for category, is_refusal in zip(dataset.categories, per_prompt):
            grouped.setdefault(category, []).append(is_refusal)
        for category, results in grouped.items():
            categories[category] = {
                "refusals": sum(results),
                "n": len(results),
                "rate": sum(results) / len(results),
            }

    return {
        "n": len(dataset.prompts),
        "refusals": len(flagged),
        "rate": len(flagged) / max(len(dataset.prompts), 1),
        "flags": flagged,
        "categories": categories,
        "elapsed_seconds": round(elapsed, 1),
    }


def compute_logprobs(model: Model, prompts) -> torch.Tensor:
    logits = model.get_logits_batched(prompts)
    return F.log_softmax(logits.float(), dim=-1).cpu()


def robust_kl_divergence(
    baseline_logprobs: torch.Tensor,
    logprobs: torch.Tensor,
) -> tuple[float, int]:
    """KL(baseline || model) with non-finite terms masked out."""
    target = baseline_logprobs.float()
    term = target.exp() * (target - logprobs.float().to(target.device))
    finite = torch.isfinite(term)
    dropped = int((~finite).sum().item())
    return float(term[finite].sum().item() / term.shape[0]), dropped


def print_refusals(flags: list[dict[str, Any]], limit: int, dataset: str) -> None:
    for flag in flags[:limit]:
        print()
        print(
            f"[bold cyan]#{flag['index']} [{dataset}] "
            f"marker={escape(repr(flag['marker']))}[/]"
        )
        print(f"[yellow]PROMPT:[/] {escape(flag['prompt'][:300])}")
        print(f"[green]RESPONSE:[/] {escape(flag['response'][:800])}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate abliterated models")
    parser.add_argument(
        "--baseline-model",
        required=True,
        help="original model used as refusal/KL reference",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=[],
        help="additional models to evaluate against the baseline",
    )
    parser.add_argument(
        "--datasets",
        default="mlabonne",
        help="comma-separated: mlabonne,jbb,tulu,nous",
    )
    parser.add_argument(
        "--prompts",
        action="append",
        default=[],
        help="extra plain-text prompt file to evaluate (repeatable)",
    )
    parser.add_argument("--limit", type=int, default=0, help="prompts per dataset (0 = all)")
    parser.add_argument(
        "--revision",
        action="append",
        default=[],
        metavar="DATASET=COMMIT",
        help="pin a dataset revision, e.g. jbb=<commit> (repeatable)",
    )
    parser.add_argument("--config", help="heretic TOML for markers and KL prompts")
    parser.add_argument("--markers-file", help="plain-text file, one marker per line")
    parser.add_argument(
        "--keep-topic-markers",
        action="store_true",
        help="do not strip illegal/harmful/etc. from the marker list",
    )
    parser.add_argument("--no-kl", action="store_true", help="skip KL divergence")
    parser.add_argument("--print-refusals", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--dtype", action="append")
    parser.add_argument("--quantization", choices=["none", "bnb_4bit"])
    parser.add_argument("--device-map")
    parser.add_argument("--response-prefix")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="system prompt for evaluation (default: from config.toml)",
    )
    parser.add_argument("--json", help="path for the JSON report")
    args = parser.parse_args()

    torch.set_grad_enabled(False)

    dataset_names = [name.strip() for name in args.datasets.split(",") if name.strip()]

    revisions: dict[str, str] = {}
    for entry in args.revision:
        if "=" not in entry:
            raise SystemExit(f"--revision expects DATASET=COMMIT, got {entry!r}")
        name, commit = entry.split("=", 1)
        revisions[name.strip()] = commit.strip()

    eval_settings = load_settings(
        config=args.config,
        model=args.baseline_model,
        values={"system_prompt": args.system_prompt}
        if args.system_prompt is not None
        else None,
    )
    if args.markers_file:
        with open(args.markers_file, encoding="utf-8") as handle:
            markers = [line.strip() for line in handle if line.strip()]
    else:
        markers = get_keyword_markers(
            eval_settings, strip_topic_markers=not args.keep_topic_markers
        )
    print(f"* Refusal markers ({len(markers)}): {markers}")

    refusal_spec, kl_spec = get_scorer_prompt_specs(eval_settings)
    kl_prompts = load_prompts(eval_settings, kl_spec)
    if args.limit:
        kl_prompts = kl_prompts[: args.limit]
    print(f"* KL reference prompts: {len(kl_prompts)}")

    datasets: list[EvalDataset] = []
    dataset_revisions: dict[str, str | None] = {}
    for name in dataset_names:
        if name == "mlabonne":
            spec = refusal_spec
            override = revisions.get("mlabonne")
            if override:
                if isinstance(spec, list):
                    spec = [
                        item.model_copy(update={"commit": override})
                        for item in spec
                    ]
                else:
                    spec = spec.model_copy(update={"commit": override})
            pinned = load_prompts(eval_settings, spec)
            if args.limit:
                pinned = pinned[: args.limit]
            datasets.append(
                EvalDataset(
                    name="mlabonne",
                    prompts=[prompt.user for prompt in pinned],
                    systems=[prompt.system for prompt in pinned],
                )
            )
            if isinstance(spec, list):
                commits = [item.commit for item in spec]
                dataset_revisions["mlabonne"] = override or (
                    ",".join(str(commit) for commit in commits)
                    if any(commits)
                    else None
                )
            else:
                dataset_revisions["mlabonne"] = override or spec.commit
        else:
            revision = revisions.get(name)
            datasets.append(load_eval_dataset(name, args.limit, revision))
            dataset_revisions[name] = revision
    datasets.extend(load_text_prompts(path, args.limit) for path in args.prompts)
    for dataset in datasets:
        dataset_revisions.setdefault(dataset.name, None)

    print(
        "* Evaluation datasets: "
        + ", ".join(f"{dataset.name} ({len(dataset.prompts)})" for dataset in datasets)
    )
    print(
        "* Dataset revisions: "
        + ", ".join(
            f"{name}={revision or 'main'}" for name, revision in dataset_revisions.items()
        )
    )

    model_paths = [args.baseline_model, *args.models]
    results: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "baseline_model": args.baseline_model,
        "markers": markers,
        "strip_topic_markers": not args.keep_topic_markers,
        "kl_prompts": len(kl_prompts),
        "datasets": {dataset.name: len(dataset.prompts) for dataset in datasets},
        "dataset_revisions": dataset_revisions,
        "models": {},
        "environment": environment_report(),
    }

    baseline_logprobs = None

    for model_path in model_paths:
        print()
        print(f"[bold]=== Evaluating {escape(model_path)} ===[/]")
        model = load_model(model_path, args)

        model_result: dict[str, Any] = {
            "refusals": {},
            "kl_divergence": None,
        }
        for dataset in datasets:
            print(f"* {dataset.name}: generating {len(dataset.prompts)} responses...")
            dataset_result = evaluate_refusals(model, dataset, markers)
            model_result["refusals"][dataset.name] = dataset_result
            print(
                f"  -> {dataset_result['refusals']}/{dataset_result['n']} "
                f"({dataset_result['rate'] * 100:.1f}%) in "
                f"{format_duration(dataset_result['elapsed_seconds'])}"
            )
            if args.print_refusals:
                print_refusals(
                    dataset_result["flags"], args.print_refusals, dataset.name
                )
            top_categories = sorted(
                dataset_result["categories"].items(),
                key=lambda item: item[1]["rate"],
                reverse=True,
            )[:5]
            for category, category_result in top_categories:
                print(
                    f"    {category}: {category_result['refusals']}/"
                    f"{category_result['n']} ({category_result['rate'] * 100:.0f}%)"
                )

        if not args.no_kl and kl_prompts:
            print(f"* KL divergence over {len(kl_prompts)} prompts...")
            if model_path == args.baseline_model:
                baseline_logprobs = compute_logprobs(model, kl_prompts)
                model_result["kl_divergence"] = 0.0
                print("  -> baseline, KL = 0 by definition")
            elif baseline_logprobs is None:
                print("  -> skipped (no baseline logprobs)")
            else:
                logprobs = compute_logprobs(model, kl_prompts)
                kl_divergence, dropped = robust_kl_divergence(
                    baseline_logprobs, logprobs
                )
                model_result["kl_divergence"] = kl_divergence
                model_result["kl_masked_positions"] = dropped
                print(f"  -> KL divergence: {kl_divergence:.4f}")

        results["models"][model_path] = model_result

        del model
        gc.collect()
        empty_cache()

    print()
    table = Table(title="Evaluation summary")
    table.add_column("Model", min_width=24)
    for dataset in datasets:
        table.add_column(dataset.name, justify="right")
    if not args.no_kl:
        table.add_column("KL", justify="right")

    for model_path, model_result in results["models"].items():
        row = [escape(model_path)]
        for dataset in datasets:
            dataset_result = model_result["refusals"][dataset.name]
            row.append(
                f"{dataset_result['refusals']}/{dataset_result['n']} "
                f"({dataset_result['rate'] * 100:.1f}%)"
            )
        if not args.no_kl:
            kl_divergence = model_result["kl_divergence"]
            row.append("n/a" if kl_divergence is None else f"{kl_divergence:.4f}")
        table.add_row(*row)
    print(table)

    if args.json:
        write_json(args.json, results)
        print(f"\nResults written to {escape(args.json)}")


if __name__ == "__main__":
    main()
