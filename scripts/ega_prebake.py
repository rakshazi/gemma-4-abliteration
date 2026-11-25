#!/usr/bin/env python3
"""One-pass Expert-Granular Abliteration (EGA) prebake for Gemma 4 MoE models."""

from __future__ import annotations

import argparse
import math
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from heretic.progress import patch_tqdm

patch_tqdm()

import torch
import transformers
from rich.markup import escape
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors

from common import (
    biproject_experts,
    biproject_weight_2d,
    collect_router_means,
    compute_refusal_directions,
    environment_report,
    get_dense_weight,
    get_expert_down_proj,
    get_prompt_specs,
    load_prompt_pair,
    load_settings,
    orthogonalize_directions,
    refusal_projection,
    resolve_response_prefix,
    select_layers,
    sha256_of_tensor,
    write_json,
)
from heretic.config import QuantizationMethod
from heretic.model import Model
from heretic.system import empty_cache
from heretic.utils import format_duration, print


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expert-Granular Abliteration prebake for Gemma 4 MoE models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="HF model ID or local path")
    parser.add_argument("--output", help="directory to write the prebaked model to")
    parser.add_argument(
        "--config",
        help="heretic 2.x TOML file to read model, dtype and prompt settings from",
    )
    parser.add_argument(
        "--parts",
        default="experts",
        choices=["dense", "experts", "all"],
        help="which weights to abliterate",
    )
    parser.add_argument(
        "--layers",
        default="all",
        help='layer selection: "all", "top:70" or a comma-separated list',
    )
    parser.add_argument("--scale", type=float, default=1.0, help="dense ablation scale")
    parser.add_argument(
        "--expert-scale", type=float, default=1.0, help="expert ablation scale"
    )
    parser.add_argument(
        "--winsorize", type=float, default=0.995, help="residual winsorization quantile"
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=0,
        help="truncate both prompt sets to N prompts (0 = all)",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--dtype", action="append", help="model dtype to try (repeatable)"
    )
    parser.add_argument("--quantization", choices=["none", "bnb_4bit"], default=None)
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--response-prefix", default=None)
    parser.add_argument(
        "--directions",
        help="reuse a precomputed ega-directions.safetensors instead of recomputing",
    )
    parser.add_argument(
        "--save-directions",
        help="path for the directions file (default: <output>/ega-directions.safetensors)",
    )
    parser.add_argument(
        "--expert-select",
        choices=["all", "routing"],
        default="all",
        help="select experts uniformly or by harmful-vs-harmless routing weight",
    )
    parser.add_argument(
        "--expert-top-pct",
        type=float,
        default=100,
        help="percentage of experts per layer to abliterate (with --expert-select routing)",
    )
    parser.add_argument(
        "--expert-chunk", type=int, default=16, help="experts processed per chunk"
    )
    parser.add_argument("--report", help="path for the JSON report")
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument(
        "--show", type=int, default=0, help="print N sample responses before and after"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="compute and report without saving weights",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace an existing output directory"
    )
    return parser.parse_args()


def autodetect_batch_size(model: Model, settings, prompts) -> int:
    print()
    print("Determining optimal batch size...")
    best_batch_size = 1
    best_performance = -1.0
    batch_size = 1
    while batch_size <= settings.max_batch_size:
        batch = (list(prompts) * math.ceil(batch_size / len(prompts)))[:batch_size]
        try:
            model.get_responses(batch)
            start = time.perf_counter()
            responses = model.get_responses(batch)
            elapsed = time.perf_counter() - start
            tokens = sum(len(model.tokenizer.encode(r)) for r in responses)
            performance = tokens / max(elapsed, 1e-6)
            print(f"* Batch size {batch_size}: {performance:.0f} tokens/s")
            if performance > best_performance:
                best_batch_size = batch_size
                best_performance = performance
        except Exception:
            break
        batch_size *= 2
    print(f"* Chosen batch size: {best_batch_size}")
    return best_batch_size


def sample_responses(model: Model, prompts, count: int) -> list[tuple[str, str]]:
    if count <= 0:
        return []
    subset = list(prompts)[:count]
    responses = model.get_responses_batched(subset, skip_special_tokens=True)
    return [(prompt.user, response) for prompt, response in zip(subset, responses)]


def print_samples(title: str, samples) -> None:
    if not samples:
        return
    print()
    print(f"[bold]{title}[/]")
    for index, (prompt, response) in enumerate(samples, 1):
        response = " ".join(response.split())[:400]
        print(f"  {index}. [yellow]{escape(prompt[:160])}[/]")
        print(f"     [green]{escape(response)}[/]")


def apply_matrix(
    weight: torch.Tensor,
    direction: torch.Tensor,
    scale: float,
) -> tuple[float, float]:
    """Abliterate one 2D matrix in place; returns (norm error, projection ratio)."""
    original_norms = weight.float().norm(dim=1)
    before_projection = refusal_projection(weight, direction)

    updated = biproject_weight_2d(weight, direction, scale)
    updated_norms = updated.float().norm(dim=1)
    norm_error = (
        (updated_norms - original_norms).abs() / original_norms.clamp(min=1e-8)
    ).max().item()

    after_projection = refusal_projection(updated, direction)
    ratio = after_projection.norm().item() / max(before_projection.norm().item(), 1e-12)

    weight.data.copy_(updated)
    return float(norm_error), float(ratio)


def apply_experts(
    expert_weights: torch.Tensor,
    direction: torch.Tensor,
    scale: float,
    expert_indices: torch.Tensor | None,
    chunk_size: int,
) -> tuple[float, float, int]:
    """Abliterate expert [E, out, in] slices in place."""
    direction_f32 = direction.float().to(expert_weights.device)
    num_experts, rows, inner = expert_weights.shape

    original_norms = torch.empty(
        (num_experts, rows), dtype=torch.float32, device=expert_weights.device
    )
    before_projection = torch.empty(
        (num_experts, inner), dtype=torch.float32, device=expert_weights.device
    )
    for start in range(0, num_experts, chunk_size):
        chunk = expert_weights[start : start + chunk_size].float()
        original_norms[start : start + chunk_size] = chunk.norm(dim=2)
        before_projection[start : start + chunk_size] = torch.einsum(
            "o,eoi->ei", direction_f32, chunk
        )
        del chunk

    if expert_indices is None:
        updated = biproject_experts(expert_weights, direction, scale, chunk_size)
        modified = num_experts
    else:
        updated = expert_weights.clone()
        for index in expert_indices.tolist():
            updated[index] = biproject_weight_2d(expert_weights[index], direction, scale)
        modified = len(expert_indices)

    updated_norms = torch.empty_like(original_norms)
    after_projection = torch.empty_like(before_projection)
    for start in range(0, num_experts, chunk_size):
        chunk = updated[start : start + chunk_size].float()
        updated_norms[start : start + chunk_size] = chunk.norm(dim=2)
        after_projection[start : start + chunk_size] = torch.einsum(
            "o,eoi->ei", direction_f32, chunk
        )
        del chunk

    norm_error = (updated_norms - original_norms).abs() / original_norms.clamp(min=1e-8)
    if expert_indices is not None:
        norm_error = norm_error[expert_indices]
        after_projection = after_projection[expert_indices]
        before_projection = before_projection[expert_indices]

    ratio = after_projection.norm().item() / max(before_projection.norm().item(), 1e-12)
    expert_weights.data.copy_(updated)
    return float(norm_error.max().item()), float(ratio), modified


def main() -> None:
    args = parse_args()

    if not args.dry_run and not args.output:
        raise SystemExit("--output is required unless --dry-run is given")

    output = Path(args.output) if args.output else None
    if output is not None and output.exists():
        if not output.is_dir():
            raise SystemExit(f"output path {output} exists and is not a directory")
        if any(output.iterdir()) and not args.overwrite:
            raise SystemExit(
                f"output directory {output} is not empty (use --overwrite to replace)"
            )

    values: dict = {}
    if args.dtype:
        values["dtypes"] = args.dtype
    if args.quantization:
        values["quantization"] = QuantizationMethod(args.quantization)
    if args.device_map:
        values["device_map"] = args.device_map
    if args.batch_size is not None:
        values["batch_size"] = args.batch_size
    if args.response_prefix is not None:
        values["response_prefix"] = args.response_prefix

    settings = load_settings(model=args.model, config=args.config, values=values)

    if settings.quantization != QuantizationMethod.NONE:
        raise SystemExit(
            "EGA prebake modifies raw weights and cannot run on a quantized model; "
            'set quantization = "none"'
        )

    torch.set_grad_enabled(False)
    transformers.logging.set_verbosity_error()

    started = time.perf_counter()

    print()
    print("[bold]Expert-Granular Abliteration prebake[/]")
    print(f"* Model: {escape(args.model)}")
    print(f"* Parts: [bold]{args.parts}[/], layers: [bold]{args.layers}[/]")
    print(f"* Dense scale: {args.scale}, expert scale: {args.expert_scale}")
    print(f"* Winsorization: {args.winsorize}")

    model = Model(settings)
    layers = model.get_layers()
    n_layers = len(layers)

    expert_param = get_expert_down_proj(layers[0])
    if expert_param is None:
        if args.parts in {"experts", "all"}:
            print("* [yellow]No fused 3D expert down_proj found; running dense-only[/]")
            args.parts = "dense"
        n_experts = 0
        expert_shape = None
    else:
        n_experts = expert_param.shape[0]
        expert_shape = list(expert_param.shape)
        print(
            f"* MoE: [bold]{n_experts}[/] experts per layer, "
            f"expert down_proj {expert_shape}"
        )

    good_prompts, bad_prompts = load_prompt_pair(settings, args.max_prompts or None)
    good_spec, bad_spec = get_prompt_specs(settings)

    def dump_spec(spec):
        if isinstance(spec, list):
            return [item.model_dump() for item in spec]
        return spec.model_dump()

    print(f"* Prompts: {len(good_prompts)} good, {len(bad_prompts)} bad")

    if settings.batch_size == 0:
        settings.batch_size = autodetect_batch_size(model, settings, good_prompts)

    settings.response_prefix = resolve_response_prefix(
        model, settings, good_prompts, bad_prompts
    )

    samples_before = sample_responses(model, bad_prompts, args.show)

    directions_path = args.directions
    if directions_path:
        print()
        print(f"Loading precomputed directions from {escape(directions_path)}...")
        payload = load_safetensors(directions_path)
        refusal_directions = payload["directions"].float()
        raw_directions = payload.get("raw_directions", refusal_directions)
        good_means = payload["good_means"].float()
        bad_means = payload["bad_means"].float()
        if refusal_directions.shape[0] != n_layers + 1:
            raise SystemExit(
                f"directions have {refusal_directions.shape[0]} rows but the model "
                f"has {n_layers} layers (+ embeddings)"
            )
    else:
        raw_directions, good_means, bad_means = compute_refusal_directions(
            model, good_prompts, bad_prompts, args.winsorize
        )
        refusal_directions = orthogonalize_directions(raw_directions, good_means)

    selected_layers, qualities = select_layers(
        raw_directions, good_means, bad_means, args.layers, n_layers
    )
    print()
    print(
        f"Selected [bold]{len(selected_layers)}[/]/{n_layers} layers: "
        f"{selected_layers}"
    )
    print(
        "* Layer quality: "
        + ", ".join(f"L{i}={qualities[i]:.4f}" for i in selected_layers)
    )

    expert_indices_per_layer: dict[int, torch.Tensor | None] = {}
    if args.parts in {"experts", "all"}:
        if args.expert_select == "routing" and args.expert_top_pct < 100:
            harmful_routing = collect_router_means(model, bad_prompts, n_layers)
            harmless_routing = collect_router_means(model, good_prompts, n_layers)
            safety_scores = harmful_routing - harmless_routing
            n_select = max(1, round(n_experts * args.expert_top_pct / 100))
            for layer_index in selected_layers:
                ranked = safety_scores[layer_index].argsort(descending=True)
                expert_indices_per_layer[layer_index] = ranked[:n_select]
            print(
                f"* Routing-based expert selection: top {n_select}/{n_experts} per layer"
            )
        else:
            for layer_index in selected_layers:
                expert_indices_per_layer[layer_index] = None

    print()
    print(f"[bold]Applying EGA ablation (dry-run: {args.dry_run})[/]")

    dense_modified = 0
    expert_slices_modified = 0
    norm_errors: list[float] = []
    projection_ratios: list[float] = []
    per_layer_ratios: dict[str, float] = {}

    for layer_index in selected_layers:
        layer = layers[layer_index]
        direction = refusal_directions[layer_index + 1]
        layer_ratios = []

        if args.parts in {"dense", "all"}:
            for name, module in (
                ("o_proj", layer.self_attn.o_proj),
                ("mlp.down_proj", layer.mlp.down_proj),
            ):
                weight = get_dense_weight(module)
                if weight is None:
                    print(f"* [yellow]L{layer_index:02d} {name}: no weight found[/]")
                    continue
                norm_error, ratio = apply_matrix(weight, direction, args.scale)
                norm_errors.append(norm_error)
                projection_ratios.append(ratio)
                layer_ratios.append(ratio)
                dense_modified += 1
                print(
                    f"* L{layer_index:02d} {name}: rows={weight.shape[0]} "
                    f"norm_err={norm_error:.2e} proj_ratio={ratio:.4f}"
                )

        if args.parts in {"experts", "all"}:
            experts = get_expert_down_proj(layer)
            if experts is None:
                print(f"* [yellow]L{layer_index:02d} experts: no down_proj found[/]")
                continue
            norm_error, ratio, modified = apply_experts(
                experts,
                direction,
                args.expert_scale,
                expert_indices_per_layer.get(layer_index),
                args.expert_chunk,
            )
            norm_errors.append(norm_error)
            projection_ratios.append(ratio)
            layer_ratios.append(ratio)
            expert_slices_modified += modified
            print(
                f"* L{layer_index:02d} experts.down_proj: {modified}/{n_experts} experts "
                f"norm_err={norm_error:.2e} proj_ratio={ratio:.4f}"
            )

        if layer_ratios:
            per_layer_ratios[str(layer_index)] = round(max(layer_ratios), 6)

    empty_cache()

    samples_after = sample_responses(model, bad_prompts, args.show)
    print_samples("Before ablation", samples_before)
    print_samples("After ablation", samples_after)

    elapsed = time.perf_counter() - started
    report = {
        "tool": "ega_prebake",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "output": args.output,
        "dry_run": args.dry_run,
        "parts": args.parts,
        "layers": args.layers,
        "selected_layers": selected_layers,
        "layer_quality": {str(k): round(v, 6) for k, v in qualities.items()},
        "n_layers": n_layers,
        "n_experts": n_experts,
        "expert_shape": expert_shape,
        "scale": args.scale,
        "expert_scale": args.expert_scale,
        "winsorize": args.winsorize,
        "expert_select": args.expert_select,
        "expert_top_pct": args.expert_top_pct,
        "prompt_counts": {"good": len(good_prompts), "bad": len(bad_prompts)},
        "prompt_specs": {
            "good": dump_spec(good_spec),
            "bad": dump_spec(bad_spec),
        },
        "system_prompt": settings.system_prompt,
        "response_prefix": settings.response_prefix,
        "batch_size": settings.batch_size,
        "directions_sha256": sha256_of_tensor(refusal_directions),
        "raw_directions_sha256": sha256_of_tensor(raw_directions),
        "dense_modified": dense_modified,
        "expert_slices_modified": expert_slices_modified,
        "checks": {
            "row_norm_max_rel_err": max(norm_errors) if norm_errors else None,
            "projection_ratio_max": max(projection_ratios)
            if projection_ratios
            else None,
            "projection_ratio_mean": (
                sum(projection_ratios) / len(projection_ratios)
                if projection_ratios
                else None
            ),
            "projection_ratio_per_layer": per_layer_ratios,
        },
        "directions_file": str(args.save_directions or ""),
        "elapsed_seconds": round(elapsed, 1),
        "environment": environment_report(),
    }

    print()
    print("[bold]Checks[/]")
    print(f"* Row norm max relative error: {report['checks']['row_norm_max_rel_err']}")
    print(f"* Refusal projection ratio max: {report['checks']['projection_ratio_max']}")
    print(f"* Dense matrices modified: {dense_modified}")
    print(f"* Expert slices modified: {expert_slices_modified}")
    print(f"* Elapsed: {format_duration(elapsed)}")

    if args.dry_run:
        report_path = args.report or os.path.join(
            "reports",
            f"ega-prebake-dryrun-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json",
        )
        write_json(report_path, report)
        print(
            f"\n[yellow]Dry run: model not saved. Report written to "
            f"{escape(report_path)}[/]"
        )
        return

    output = Path(args.output)
    temporary = output.with_name(output.name + ".tmp")
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True, exist_ok=True)

    print()
    print(f"Saving prebaked model to {escape(str(output))}...")
    model.model.save_pretrained(temporary, max_shard_size=args.max_shard_size)
    model.tokenizer.save_pretrained(temporary)
    if model.processor is not None:
        model.processor.save_pretrained(temporary)

    directions_file = args.save_directions or str(
        temporary / "ega-directions.safetensors"
    )
    save_safetensors(
        {
            "directions": refusal_directions.contiguous().cpu(),
            "raw_directions": raw_directions.contiguous().cpu(),
            "good_means": good_means.contiguous().cpu(),
            "bad_means": bad_means.contiguous().cpu(),
        },
        directions_file,
    )

    if output.exists():
        shutil.rmtree(output)
    temporary.rename(output)

    def final_location(path: str) -> str:
        """Map a path inside the temporary directory to its post-rename location."""
        absolute = os.path.abspath(path)
        temporary_absolute = os.path.abspath(str(temporary))
        if absolute == temporary_absolute or absolute.startswith(
            temporary_absolute + os.sep
        ):
            return os.path.join(
                str(output), os.path.relpath(absolute, temporary_absolute)
            )
        return str(path)

    directions_file = final_location(directions_file)
    report["directions_file"] = directions_file

    report_path = final_location(
        args.report or str(temporary / "ega-report.json")
    )
    write_json(report_path, report)

    print(f"[bold green]Saved to {escape(str(output))}[/]")
    print(f"* Report: {escape(report_path)}")
    print(f"* Directions: {escape(directions_file)}")


if __name__ == "__main__":
    main()
