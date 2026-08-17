# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A declaration must match the cache that actually gets allocated.

These tests build the real `transformers` cache objects the models build and
compare their allocated bytes against what the spec claims, so a wrong
declaration fails here rather than being believed. An earlier version of this
file asserted hand-typed constants against each other, which could not fail no
matter what any model declared.
"""

import ast
import pathlib

import pytest
import torch
from transformers import Qwen2Config, WhisperConfig
from transformers.cache_utils import DynamicCache, StaticCache

from vllm_omni.model_executor.models.model_local_kv import (
    HasModelLocalKV,
    ModelLocalKVScope,
    ModelLocalKVSpec,
    collect_model_local_kv_specs,
    spec_from_hf_config,
    total_declared_bytes,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

MODELS_DIR = pathlib.Path(__file__).resolve().parents[3] / "vllm_omni" / "model_executor" / "models"


def _cache_bytes(cache) -> int:
    """Sum the real allocated bytes of a transformers cache."""
    total = 0
    for layer in cache.layers:
        for tensor in (layer.keys, layer.values):
            if tensor is not None:
                total += tensor.numel() * tensor.element_size()
    return total


def test_declaration_matches_a_real_static_cache():
    """ming's talker: one decode step costs the whole declared extent.

    Built with the same arguments as `MingAudioGenerator._init_kv_cache`.
    transformers v5 initializes `StaticLayer` lazily -- `keys` is None until a
    layer is first written -- but that first write allocates all
    `max_cache_len` positions at once rather than growing. A prefill touches
    every layer, so the declared number is reached on the first step of any
    generation regardless of how short the utterance is.
    """
    layers, kv_heads, head_dim = 4, 2, 32
    config = Qwen2Config(
        num_hidden_layers=layers,
        num_attention_heads=8,
        num_key_value_heads=kv_heads,
        hidden_size=256,
        intermediate_size=512,
        vocab_size=1000,
    )
    spec = spec_from_hf_config(
        config,
        name="talker_llm_static_cache",
        dtype=torch.float16,
        physical_capacity_positions=2048,
        capacity_source="hardcoded max_cache_len=2048",
        scope=ModelLocalKVScope.INVOCATION,
        batch_capacity=1,
        max_live_instances=1,
    )
    cache = StaticCache(
        config=config,
        max_batch_size=1,
        max_cache_len=2048,
        device="cpu",
        dtype=torch.float16,
    )
    assert _cache_bytes(cache) == 0, "expected v5 lazy init; a preallocating build changes the claim below"

    # A single one-position write per layer, i.e. the cheapest possible step.
    for layer_idx in range(layers):
        keys = torch.zeros(1, kv_heads, 1, head_dim, dtype=torch.float16)
        cache.update(keys, keys.clone(), layer_idx)

    assert spec.bytes_per_instance == _cache_bytes(cache)


def test_declaration_matches_a_real_dynamic_cache_at_capacity():
    """The growing caches: declared bytes are the ceiling they reach when full.

    Fills a DynamicCache to the declared position count and compares. This is
    what makes `physical_capacity_positions` meaningful -- if the field were
    read as a logical sequence position rather than tensor extent, this fails.
    """
    layers, kv_heads, head_dim, positions = 4, 2, 32, 11
    config = Qwen2Config(
        num_hidden_layers=layers,
        num_attention_heads=kv_heads,
        num_key_value_heads=kv_heads,
        hidden_size=kv_heads * head_dim,
        intermediate_size=128,
        vocab_size=1000,
    )
    spec = spec_from_hf_config(
        config,
        name="local_transformer",
        dtype=torch.bfloat16,
        physical_capacity_positions=positions,
        capacity_source="group_size + max(delay_pattern)",
        scope=ModelLocalKVScope.INVOCATION,
        batch_capacity=1,
        max_live_instances=1,
    )

    cache = DynamicCache()
    for layer_idx in range(layers):
        keys = torch.zeros(1, kv_heads, positions, head_dim, dtype=torch.bfloat16)
        cache.update(keys, torch.zeros_like(keys), layer_idx)

    assert spec.bytes_per_instance == _cache_bytes(cache)


def test_multiplicity_is_what_graph_capture_costs():
    """Retained copies dominate, and only max_live_instances can say so.

    Both Qwen3-TTS and MiMo keep one cache per captured graph bucket alive for
    the process lifetime. A single preallocated/grows flag cannot express a
    cache that is per-invocation but replicated 261 times.
    """
    config = Qwen2Config(
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_size=64,
        intermediate_size=128,
        vocab_size=1000,
    )
    kwargs = dict(
        config=config,
        name="graph_pool",
        dtype=torch.bfloat16,
        physical_capacity_positions=11,
        capacity_source="test",
        scope=ModelLocalKVScope.MODEL,
        batch_capacity=1,
    )
    one = spec_from_hf_config(max_live_instances=1, **kwargs)
    captured = spec_from_hf_config(max_live_instances=261, **kwargs)

    assert captured.bytes_per_instance == one.bytes_per_instance
    assert captured.max_live_bytes == 261 * one.max_live_bytes


def test_whisper_geometry_needs_encoder_naming():
    """Encoder-decoder configs do not use the Qwen attribute names.

    MiniCPM-o passes its geometry explicitly for this reason. Guarding it here
    so a later "simplification" to pure config sniffing fails loudly.
    """
    config = WhisperConfig(encoder_layers=4, encoder_attention_heads=8, d_model=256)
    spec = spec_from_hf_config(
        config,
        name="whisper_encoder_self_attn",
        dtype=torch.bfloat16,
        layers=config.encoder_layers,
        kv_heads=config.encoder_attention_heads,
        head_dim=config.d_model // config.encoder_attention_heads,
        physical_capacity_positions=1500,
        capacity_source="embed_positions rows",
        scope=ModelLocalKVScope.SESSION,
        batch_capacity=1,
        max_live_instances=1,
    )
    assert (spec.layers, spec.kv_heads, spec.head_dim) == (4, 8, 32)


def test_spec_from_hf_config_raises_rather_than_guessing():
    """A missing attribute must not silently become a wrong number."""

    class Empty:
        pass

    with pytest.raises(ValueError, match="layer count"):
        spec_from_hf_config(
            Empty(),
            name="x",
            dtype=torch.float16,
            physical_capacity_positions=1,
            capacity_source="",
            scope=ModelLocalKVScope.REQUEST,
        )


def _spec(**overrides) -> ModelLocalKVSpec:
    kwargs = dict(
        name="c",
        layers=2,
        kv_heads=2,
        head_dim=4,
        dtype=torch.float16,
        physical_capacity_positions=8,
        capacity_source="test",
        scope=ModelLocalKVScope.REQUEST,
        batch_capacity=1,
        max_live_instances=1,
    )
    kwargs.update(overrides)
    return ModelLocalKVSpec(**kwargs)


class _Owner(torch.nn.Module):
    def model_local_kv_specs(self):
        return [_spec()]


def test_collector_finds_owners_nested_in_the_module_tree():
    """The cache owner is never the registered model, so the walk is the point."""

    class Mid(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = _Owner()

    class Top(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mid = Mid()

    model = Top()
    collected = collect_model_local_kv_specs(model)
    assert [path for path, _ in collected] == ["mid.inner"]
    # 2 layers * 2 kv * 4 hd * 8 pos * 1 batch * 2 bytes * 2 (K and V) = 512
    assert total_declared_bytes(model) == 512


def test_collector_sees_through_a_non_module_wrapper():
    """The runner's `self.model` is not always the model.

    `vllm.compilation.cuda_graph.CUDAGraphWrapper` is a plain callable holding
    `.runnable`, not an nn.Module, so a collector that goes straight to
    `named_modules()` finds nothing and every declaration silently reports
    zero. Caught on a real Qwen3-TTS boot, where the runner logged
    `model=CUDAGraphWrapper specs=0`.
    """

    class Wrapper:  # mirrors CUDAGraphWrapper's shape, not an nn.Module
        def __init__(self, runnable):
            self.runnable = runnable

    class Top(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.owner = _Owner()

    wrapped = Wrapper(Top())
    assert not hasattr(wrapped, "named_modules")
    assert [path for path, _ in collect_model_local_kv_specs(wrapped)] == ["owner"]
    assert total_declared_bytes(wrapped) == 512


def test_collector_tolerates_a_wrapper_chain_that_never_terminates():
    class SelfWrapper:
        pass

    node = SelfWrapper()
    node.runnable = node  # cycle
    assert collect_model_local_kv_specs(node) == []


def test_collector_survives_an_owner_that_raises():
    """Reporting memory must never be able to break model load."""

    class Broken(torch.nn.Module):
        def model_local_kv_specs(self):
            raise RuntimeError("checkpoint field missing")

    class Top(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.broken = Broken()
            self.ok = _Owner()

    assert total_declared_bytes(Top()) == 512


def test_concurrency_multiplies_per_request_caches_but_not_resident_ones():
    """The model declares one unit; only the engine knows how many are live.

    A per-request cache at max_num_seqs=64 costs 64x. A cache captured into the
    CUDA-graph pool is shared across all of them and costs the same at 64 as at
    1. Collapsing these into one number is wrong in one direction or the other
    depending on which cache you pick.
    """
    per_request = _spec(scope=ModelLocalKVScope.REQUEST)
    resident = _spec(scope=ModelLocalKVScope.MODEL, max_live_instances=261)

    assert per_request.scales_with_concurrency
    assert not resident.scales_with_concurrency

    assert per_request.peak_bytes(1) == per_request.max_live_bytes
    assert per_request.peak_bytes(64) == 64 * per_request.max_live_bytes
    assert resident.peak_bytes(64) == resident.max_live_bytes


def test_peak_bytes_rejects_a_meaningless_concurrency():
    with pytest.raises(ValueError, match="concurrency"):
        _spec().peak_bytes(0)


def test_total_declared_bytes_takes_concurrency():
    class Top(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.owner = _Owner()

    model = Top()
    assert total_declared_bytes(model) == 512
    assert total_declared_bytes(model, concurrency=8) == 8 * 512


def test_undeclared_models_report_zero_not_an_error():
    """Additive before every model is migrated."""
    assert total_declared_bytes(object()) == 0
    assert total_declared_bytes(torch.nn.Linear(2, 2)) == 0


def test_batch_capacity_has_no_default():
    """Defaulting it to 1 would silently under-report every batched cache."""
    with pytest.raises(TypeError):
        ModelLocalKVSpec(  # type: ignore[call-arg]
            name="x",
            layers=1,
            kv_heads=1,
            head_dim=1,
            dtype=torch.float16,
            physical_capacity_positions=1,
            capacity_source="",
            scope=ModelLocalKVScope.REQUEST,
        )


def test_protocol_is_structural():
    assert isinstance(_Owner(), HasModelLocalKV)
    assert not isinstance(torch.nn.Linear(2, 2), HasModelLocalKV)


@pytest.mark.parametrize(
    ("relative_path", "owner"),
    [
        ("qwen3_tts/segmented_graph_wrapper.py", "CUDAGraphDecoderWrapper"),
        ("qwen3_tts/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py", "Qwen3TTSTokenizerV2Decoder"),
        ("mimo_audio/mimo_audio_llm.py", "MiMoAudioLLMForConditionalGeneration"),
        ("minicpmo_4_5/minicpmo_4_5_omni_llm.py", "MiniCPMWhisperEncoder"),
        ("ming_flash_omni/talker_module.py", "MingAudioGenerator"),
        ("ming_flash_omni/ming_flash_omni_talker.py", "MingFlashOmniTalkerForConditionalGeneration"),
    ],
)
def test_known_cache_owners_still_declare(relative_path: str, owner: str):
    """Parsed rather than imported: these modules need CUDA and real weights.

    Catches the declaration being dropped in a refactor, which would silently
    take the model back to reporting zero.
    """
    tree = ast.parse((MODELS_DIR / relative_path).read_text())
    classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == owner]
    assert classes, f"{owner} not found in {relative_path}"
    methods = {child.name for cls in classes for child in cls.body if isinstance(child, ast.FunctionDef)}
    assert "model_local_kv_specs" in methods, f"{owner} no longer declares its model-local KV"
