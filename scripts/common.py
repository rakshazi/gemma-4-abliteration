"""Shared helpers for Gemma 4 abliteration on top of heretic 2.x."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F
from rich.markup import escape

from heretic.config import DatasetSpecification, Settings, SingleDatasetSpecification
from heretic.model import Model
from heretic.utils import Prompt, load_prompts, print

# Default KeywordRate markers that match disclaimers rather than refusals.
TOPIC_MARKERS = frozenset(
    {
        "violat",
        "prohibit",
        "illegal",
        "harmful",
        "inappropriate",
        "unethical",
        "ethical boundaries",
    }
)

DEFAULT_GOOD_PROMPTS = {
    "dataset": "mlabonne/harmless_alpaca",
    "split": "train[:400]",
    "column": "text",
}

DEFAULT_BAD_PROMPTS = {
    "dataset": "mlabonne/harmful_behaviors",
    "split": "train[:400]",
    "column": "text",
}

DEFAULT_EVAL_PROMPTS = {
    "dataset": "mlabonne/harmful_behaviors",
    "split": "test[:100]",
    "column": "text",
}

DEFAULT_KL_PROMPTS = {
    "dataset": "mlabonne/harmless_alpaca",
    "split": "test[:100]",
    "column": "text",
}

GEMMA4_COT_BLOCKS = [("<|channel>thought\n", "<channel|>")]


@contextmanager
def clean_argv():
    """Hide our argparse argv from pydantic-settings' CLI source."""
    real_argv = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        yield
    finally:
        sys.argv = real_argv


def load_settings(
    *,
    model: str | None = None,
    config: str | None = None,
    values: dict[str, Any] | None = None,
) -> Settings:
    """Build heretic Settings from an optional TOML config plus init overrides."""
    old_config = os.environ.get("HERETIC_CONFIG_FILE")
    if config is not None:
        config = os.path.abspath(config)
        if not os.path.isfile(config):
            raise SystemExit(f"config file not found: {config}")
        os.environ["HERETIC_CONFIG_FILE"] = config

    merged: dict[str, Any] = dict(values or {})
    if model is not None:
        merged["model"] = model

    try:
        with clean_argv():
            return Settings(**merged)
    finally:
        if config is not None:
            if old_config is None:
                os.environ.pop("HERETIC_CONFIG_FILE", None)
            else:
                os.environ["HERETIC_CONFIG_FILE"] = old_config


def _as_specification(value: Any, default: dict[str, Any]) -> DatasetSpecification:
    if value is None:
        value = default
    if isinstance(value, list):
        return [SingleDatasetSpecification.model_validate(item) for item in value]
    return SingleDatasetSpecification.model_validate(value)


def get_prompt_specs(
    settings: Settings,
) -> tuple[DatasetSpecification, DatasetSpecification]:
    """Read abliteration good/bad prompt specs from a heretic 2.x config."""
    from heretic.plugin import get_plugin_namespace

    raw = get_plugin_namespace(settings.model_extra, "modifier.Abliteration")
    good = _as_specification(raw.get("good_prompts"), DEFAULT_GOOD_PROMPTS)
    bad = _as_specification(raw.get("bad_prompts"), DEFAULT_BAD_PROMPTS)
    return good, bad


def get_scorer_prompt_specs(
    settings: Settings,
) -> tuple[DatasetSpecification, DatasetSpecification]:
    """Read evaluation prompt specs for refusals and KL from a heretic config."""
    from heretic.plugin import get_plugin_namespace

    keyword_rate = get_plugin_namespace(settings.model_extra, "scorer.KeywordRate")
    kl_divergence = get_plugin_namespace(settings.model_extra, "scorer.KLDivergence")
    refusal = _as_specification(keyword_rate.get("prompts"), DEFAULT_EVAL_PROMPTS)
    kl_prompts = _as_specification(kl_divergence.get("prompts"), DEFAULT_KL_PROMPTS)
    return refusal, kl_prompts


def get_keyword_markers(
    settings: Settings,
    strip_topic_markers: bool = True,
) -> list[str]:
    """Refusal markers, with the false-positive-prone topic words removed by default."""
    from heretic.plugin import get_plugin_namespace
    from heretic.scorers.keyword_rate import REFUSAL_MARKERS

    raw = get_plugin_namespace(settings.model_extra, "scorer.KeywordRate")
    markers = list(raw.get("keyword_markers") or REFUSAL_MARKERS)
    if strip_topic_markers:
        markers = [marker for marker in markers if marker not in TOPIC_MARKERS]
    return markers


def load_prompt_pair(
    settings: Settings,
    max_prompts: int | None = None,
) -> tuple[list[Prompt], list[Prompt]]:
    good_spec, bad_spec = get_prompt_specs(settings)
    good = load_prompts(settings, good_spec)
    bad = load_prompts(settings, bad_spec)
    if max_prompts:
        good = good[:max_prompts]
        bad = bad[:max_prompts]
    return good, bad


def _render_dummy_prompt(model: Model) -> str:
    messages = []
    if model.supports_system_role:
        messages.append({"role": "system", "content": "You are a helpful assistant."})
    messages.append({"role": "user", "content": "This is a dummy prompt."})
    return model.tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )


def resolve_response_prefix(
    model: Model,
    settings: Settings,
    good_prompts: Sequence[Prompt],
    bad_prompts: Sequence[Prompt],
) -> str:
    """Return the response prefix, mirroring heretic 2.x prefix detection."""
    if settings.response_prefix is not None:
        return settings.response_prefix

    blocks = list(settings.chain_of_thought_blocks) + [
        block
        for block in GEMMA4_COT_BLOCKS
        if block not in settings.chain_of_thought_blocks
    ]

    print()
    print("Checking chat template for a closed Chain-of-Thought block or an initializer...")

    dummy_prompt = _render_dummy_prompt(model)
    cot_skip_applied = False

    for cot_initializer, cot_finalizer in blocks:
        if re.search(
            rf"{re.escape(cot_initializer)}.*{re.escape(cot_finalizer)}",
            dummy_prompt,
            flags=re.DOTALL,
        ):
            print(
                f"* Chat template inserts full closed CoT block "
                f"{escape(repr(cot_initializer + cot_finalizer))}"
            )
            return ""
        if cot_initializer in dummy_prompt:
            prefix = cot_finalizer
            print(
                f"* Chat template inserts CoT initializer "
                f"{escape(repr(cot_initializer))}; prefix {escape(repr(prefix))}"
            )
            cot_skip_applied = True
            break

    if not cot_skip_applied:
        print("* None found")
        print()
        print("Checking for common response prefix...")
        test_prompts = list(good_prompts[:100]) + list(bad_prompts[:100])
        responses = model.get_responses_batched(test_prompts)
        prefix = os.path.commonprefix(responses).rstrip(" ") if responses else ""

        if prefix:
            for cot_initializer, cot_finalizer in blocks:
                if prefix.startswith(cot_initializer):
                    prefix = cot_initializer + cot_finalizer
                    cot_skip_applied = True
                    print(
                        f"* Closed Chain-of-Thought block in prefix "
                        f"{escape(repr(prefix))}"
                    )
                    break
    else:
        test_prompts = list(good_prompts[:100]) + list(bad_prompts[:100])

    if cot_skip_applied and prefix:
        print("* Rechecking with prefix...")
        previous_prefix = settings.response_prefix
        settings.response_prefix = prefix
        try:
            responses = model.get_responses_batched(test_prompts)
        finally:
            settings.response_prefix = previous_prefix
        extra = os.path.commonprefix(responses).rstrip(" ")
        if extra:
            prefix += extra

    print(f"* Prefix: {escape(repr(prefix))}" if prefix else "* None found")
    return prefix


def compute_refusal_directions(
    model: Model,
    good_prompts: Sequence[Prompt],
    bad_prompts: Sequence[Prompt],
    winsorize_quantile: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-layer refusal directions plus good/bad means, shape (n_layers + 1, hidden)."""
    print()
    print("Calculating per-layer residual means...")
    print("* Obtaining residual mean for good prompts...")
    good_means = model.get_residuals_mean(
        list(good_prompts),
        winsorization_quantile=winsorize_quantile,
    )
    print("* Obtaining residual mean for bad prompts...")
    bad_means = model.get_residuals_mean(
        list(bad_prompts),
        winsorization_quantile=winsorize_quantile,
    )

    refusal_directions = F.normalize(bad_means.float() - good_means.float(), p=2, dim=1)
    return refusal_directions, good_means.float(), bad_means.float()


def orthogonalize_directions(
    refusal_directions: torch.Tensor,
    good_means: torch.Tensor,
    passes: int = 2,
) -> torch.Tensor:
    """Project the harmless direction out of each refusal direction, then renormalize."""
    good_directions = F.normalize(good_means.float(), p=2, dim=1)
    projected = refusal_directions.float().clone()
    for _ in range(passes):
        projection = torch.sum(projected * good_directions, dim=1, keepdim=True)
        projected = projected - projection * good_directions
    return F.normalize(projected, p=2, dim=1)


def compute_layer_quality(
    refusal_dir: torch.Tensor,
    harmful_mean: torch.Tensor,
    harmless_mean: torch.Tensor,
) -> float:
    """SNR * (1 - cos) * purity, as used for layer selection in the research repo."""
    snr = refusal_dir.float().norm() / max(
        harmful_mean.float().norm().item(),
        harmless_mean.float().norm().item(),
        1e-8,
    )
    cos_sim = F.cosine_similarity(
        harmful_mean.float().unsqueeze(0),
        harmless_mean.float().unsqueeze(0),
    ).item()
    harmless_hat = F.normalize(harmless_mean.float(), dim=0)
    projection = (refusal_dir.float() @ harmless_hat) * harmless_hat
    purity = (refusal_dir.float() - projection).norm().item() / max(
        refusal_dir.float().norm().item(), 1e-8
    )
    return float(snr * (1.0 - cos_sim) * purity)


def select_layers(
    refusal_directions: torch.Tensor,
    good_means: torch.Tensor,
    bad_means: torch.Tensor,
    selection: str,
    n_layers: int,
) -> tuple[list[int], dict[int, float]]:
    """Resolve "all", "top:70" or "0,5,6" to sorted layer indices plus quality scores."""
    qualities: dict[int, float] = {}
    for layer_index in range(n_layers):
        direction_index = layer_index + 1
        qualities[layer_index] = compute_layer_quality(
            refusal_directions[direction_index],
            bad_means[direction_index],
            good_means[direction_index],
        )

    normalized = selection.strip().lower()
    if normalized in {"all", "100", "100%"}:
        return list(range(n_layers)), qualities

    if normalized.startswith("top:"):
        percentage = float(normalized.split(":", 1)[1].rstrip("%"))
        n_select = max(1, min(n_layers, round(n_layers * percentage / 100)))
        ranked = sorted(qualities, key=qualities.__getitem__, reverse=True)
        return sorted(ranked[:n_select]), qualities

    indices = [int(part) for part in normalized.replace(" ", "").split(",") if part]
    if not indices:
        raise ValueError(f"could not parse layer selection: {selection!r}")
    for index in indices:
        if not 0 <= index < n_layers:
            raise ValueError(f"layer index {index} out of range 0..{n_layers - 1}")
    return sorted(set(indices)), qualities


def biproject_weight_2d(
    weight: torch.Tensor,
    refusal_dir: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """Norm-preserving biprojected ablation of a [out, in] weight matrix."""
    weight_f32 = weight.float()
    rows = weight_f32.shape[0]
    direction = refusal_dir.float()
    if direction.numel() != rows:
        raise ValueError(
            f"refusal direction has {direction.numel()} components but weight "
            f"has {rows} rows (expected output-space direction)"
        )
    direction = F.normalize(direction.to(weight_f32.device), dim=0)

    row_norms = weight_f32.norm(dim=1, keepdim=True)
    row_dirs = weight_f32 / row_norms.clamp(min=1e-8)

    component = direction @ row_dirs
    projected = row_dirs - scale * direction.unsqueeze(1) * component.unsqueeze(0)
    projected = F.normalize(projected, dim=1)

    component = direction @ projected
    projected = projected - direction.unsqueeze(1) * component.unsqueeze(0)
    projected = F.normalize(projected, dim=1)

    return (row_norms * projected).to(weight.dtype)


def biproject_experts(
    expert_weights: torch.Tensor,
    refusal_dir: torch.Tensor,
    scale: float = 1.0,
    chunk_size: int = 16,
) -> torch.Tensor:
    """Vectorized norm-preserving ablation over a [num_experts, out, in] tensor."""
    if expert_weights.dim() != 3:
        raise ValueError(f"expected 3D expert tensor, got {tuple(expert_weights.shape)}")

    rows = expert_weights.shape[1]
    direction = refusal_dir.float()
    if direction.numel() != rows:
        raise ValueError(
            f"refusal direction has {direction.numel()} components but expert "
            f"weights have {rows} output rows"
        )
    direction = F.normalize(direction.to(expert_weights.device), dim=0)

    output = torch.empty_like(expert_weights)
    for start in range(0, expert_weights.shape[0], chunk_size):
        chunk = expert_weights[start : start + chunk_size].float()
        row_norms = chunk.norm(dim=2, keepdim=True)
        row_dirs = chunk / row_norms.clamp(min=1e-8)

        component = torch.einsum("o,coi->ci", direction, row_dirs)
        projected = row_dirs - scale * direction.view(1, -1, 1) * component.unsqueeze(1)
        projected = projected / projected.norm(dim=2, keepdim=True).clamp(min=1e-8)

        component = torch.einsum("o,coi->ci", direction, projected)
        projected = projected - direction.view(1, -1, 1) * component.unsqueeze(1)
        projected = projected / projected.norm(dim=2, keepdim=True).clamp(min=1e-8)

        output[start : start + chunk_size] = (row_norms * projected).to(
            expert_weights.dtype
        )

    return output


def refusal_projection(
    weight: torch.Tensor,
    refusal_dir: torch.Tensor,
) -> torch.Tensor:
    """r^T W flattened over all non-output dimensions."""
    direction = F.normalize(refusal_dir.float().to(weight.device), dim=0)
    return direction @ weight.float().reshape(weight.shape[0], -1)


def get_dense_weight(module: torch.nn.Module) -> torch.Tensor | None:
    """Weight tensor of an nn.Linear, transparently unwrapping PEFT LoRA."""
    if hasattr(module, "base_layer") and hasattr(module.base_layer, "weight"):
        return module.base_layer.weight
    if hasattr(module, "weight"):
        return module.weight
    return None


def get_expert_down_proj(layer: torch.nn.Module) -> torch.Tensor | None:
    """The fused 3D expert down_proj parameter of a Gemma 4 MoE layer."""
    experts = getattr(layer, "experts", None)
    if experts is None:
        return None
    down_proj = getattr(experts, "down_proj", None)
    if isinstance(down_proj, torch.Tensor) and down_proj.dim() == 3:
        return down_proj
    return None


def collect_router_means(
    model: Model,
    prompts: Sequence[Prompt],
    n_layers: int,
) -> torch.Tensor:
    """Mean router probability per layer and expert over a prompt set."""
    layers = model.get_layers()
    n_experts = 0
    for layer in layers:
        expert_down = get_expert_down_proj(layer)
        if expert_down is not None:
            n_experts = expert_down.shape[0]
            break

    totals = torch.zeros(n_layers, n_experts, dtype=torch.float64)
    counts = torch.zeros(n_layers, dtype=torch.int64)
    hooks = []

    def make_hook(layer_index: int):
        def hook(module, inputs, output):
            weights = output[0] if isinstance(output, tuple) else output
            if weights is None or weights.dim() < 2:
                return
            if weights.shape[-1] != n_experts:
                return
            mean_weights = weights.float().mean(dim=tuple(range(weights.dim() - 1)))
            totals[layer_index] += mean_weights.double().cpu()
            counts[layer_index] += 1

        return hook

    for layer_index, layer in enumerate(layers):
        router = getattr(layer, "router", None)
        if router is not None:
            hooks.append(router.register_forward_hook(make_hook(layer_index)))

    if not hooks:
        raise RuntimeError("no MoE routers found - cannot rank experts by routing")

    print()
    print(f"Collecting router statistics from {len(prompts)} prompts...")
    try:
        model.get_residuals_batched(list(prompts))
    finally:
        for handle in hooks:
            handle.remove()

    for layer_index in range(n_layers):
        if counts[layer_index] > 0:
            totals[layer_index] /= counts[layer_index]

    return totals.float()


def normalize_response(response: str) -> str:
    response = response.lower().replace("*", "").replace("\u2019", "'")
    return " ".join(response.split())


def match_refusal(response: str, markers: Iterable[str]) -> str | None:
    """Return the matched marker, "" for an empty response, or None."""
    if not response.strip():
        return ""
    normalized = normalize_response(response)
    for marker in markers:
        if marker.lower() in normalized:
            return marker
    return None


def environment_report() -> dict[str, str]:
    def version(name: str) -> str:
        try:
            return pkg_version(name)
        except PackageNotFoundError:
            return "unknown"

    return {
        "heretic-llm": version("heretic-llm"),
        "transformers": version("transformers"),
        "torch": torch.__version__,
        "accelerate": version("accelerate"),
        "peft": version("peft"),
        "datasets": version("datasets"),
    }


def sha256_of_tensor(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.contiguous().cpu().numpy().tobytes()).hexdigest()


def write_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
