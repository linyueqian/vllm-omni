# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The spec must reproduce the four measured footprints.

Each expected number below was derived from the checkpoint config and the code
that bounds the cache, not from running the model. The arithmetic lives in the
docstring of each test so a future reader can re-check it without re-deriving
where the bound comes from.
"""

import pytest
import torch

from vllm_omni.model_executor.models.model_local_kv import (
    HasModelLocalKV,
    ModelLocalKVScope,
    ModelLocalKVSpec,
    total_declared_bytes,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_qwen3_tts_codec_decoder():
    """Sliding window pins physical capacity at sliding_window - 1 = 71.

    2 * 8 layers * 16 kv_heads * 64 head_dim * 71 * 1 * 4 (fp32) = 4,653,056 B
    """
    spec = ModelLocalKVSpec(
        name="codec_decoder",
        layers=8,
        kv_heads=16,
        head_dim=64,
        dtype=torch.float32,
        physical_capacity_positions=71,
        capacity_source="sliding_window(72) - 1; all layers sliding_attention",
        scope=ModelLocalKVScope.REQUEST,
        batch_capacity=1,
        max_live_instances=1,
    )
    assert spec.bytes_per_instance == 4_653_056
    assert spec.max_live_bytes == 4_653_056


def test_mimo_local_transformer():
    """Bounded by the decode loop, not by sequence length.

    group_size(4) + max(delay_pattern)(7) = 11 steps.
    2 * 16 * 64 * 16 * 11 * 1 * 2 (bf16) = 720,896 B
    """
    spec = ModelLocalKVSpec(
        name="local_transformer",
        layers=16,
        kv_heads=64,
        head_dim=16,
        dtype=torch.bfloat16,
        physical_capacity_positions=11,
        capacity_source="group_size + max(delay_pattern)",
        scope=ModelLocalKVScope.INVOCATION,
        batch_capacity=1,
        max_live_instances=1,
    )
    assert spec.bytes_per_instance == 720_896


def test_mimo_graph_resident_copies_are_a_separate_entry():
    """The captured graphs are the real cost, and only multiplicity shows it.

    One buffer per bucket in MIMO_CUDAGRAPH_BATCH_SIZES; summing the buckets
    gives 1+2+4+6+8+16+32+64+128 = 261 batch rows retained for the process
    lifetime. A single scope flag cannot express this -- it needs multiplicity.
    """
    spec = ModelLocalKVSpec(
        name="local_transformer_graph_pool",
        layers=16,
        kv_heads=64,
        head_dim=16,
        dtype=torch.bfloat16,
        physical_capacity_positions=11,
        capacity_source="group_size + max(delay_pattern)",
        scope=ModelLocalKVScope.MODEL,
        batch_capacity=1,
        max_live_instances=261,
    )
    assert spec.max_live_bytes == 188_153_856  # 179.44 MiB
    assert spec.max_live_bytes > 100 * spec.bytes_per_instance


def test_minicpmo_whisper_encoder():
    """Bounded by encoder frames, not max_model_len.

    2 * 24 * 16 * 64 * 1500 * 1 * 2 (bf16) = 147,456,000 B
    """
    spec = ModelLocalKVSpec(
        name="whisper_encoder_self_attn",
        layers=24,
        kv_heads=16,
        head_dim=64,
        dtype=torch.bfloat16,
        physical_capacity_positions=1500,
        capacity_source="apm.embed_positions max_source_positions",
        scope=ModelLocalKVScope.SESSION,
        batch_capacity=1,
        max_live_instances=1,
    )
    assert spec.bytes_per_instance == 147_456_000


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


def test_total_declared_bytes_is_zero_for_undeclared_models():
    """Additive before every model is migrated; must not raise."""

    class Undeclared:
        pass

    assert total_declared_bytes(Undeclared()) == 0


def test_protocol_is_structural():
    class Declared:
        def model_local_kv_specs(self):
            return [
                ModelLocalKVSpec(
                    name="c",
                    layers=2,
                    kv_heads=2,
                    head_dim=4,
                    dtype=torch.float16,
                    physical_capacity_positions=8,
                    capacity_source="test",
                    scope=ModelLocalKVScope.REQUEST,
                    batch_capacity=1,
                    max_live_instances=3,
                )
            ]

    m = Declared()
    assert isinstance(m, HasModelLocalKV)
    # 2*2*2*4*8*1*2 = 512 per instance, 3 live
    assert total_declared_bytes(m) == 1536
