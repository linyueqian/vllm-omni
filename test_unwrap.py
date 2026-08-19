import torch

from vllm_omni.model_executor.models.model_local_kv import (
    Fixed,
    ModelLocalKVScope,
    ModelLocalKVSpec,
    collect_model_local_kv_specs,
)


class _Owner(torch.nn.Module):
    def model_local_kv_specs(self):
        return [
            ModelLocalKVSpec(
                name="c",
                layers=2,
                kv_heads=2,
                head_dim=4,
                dtype=torch.float32,
                physical_capacity_positions=8,
                capacity_source="test",
                scope=ModelLocalKVScope.REQUEST,
                rows=Fixed(1, because="test"),
            )
        ]


class CUDAGraphWrapperLike_NoUnwrap:
    def __init__(self, runnable):
        self.runnable = runnable

    def __getattr__(self, key):
        return getattr(self.runnable, key)


class Root(_Owner):
    pass


wrapped = CUDAGraphWrapperLike_NoUnwrap(Root())
collected = collect_model_local_kv_specs(wrapped)
print(f"Collected len: {len(collected)}")
for p, s in collected:
    print(f"Path: '{p}', Spec name: {s.name}")
