# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Batched AuK DiT steps across requests.

Requests of different text, reference and target lengths, with and without
guidance, on different grids and at different steps share one padded, masked
forward; each request's velocity must match what it gets alone.
"""

import pytest
import torch

from vllm_omni.diffusion.models.auk.auk_transformer import (
    AuKTransformer,
    _sample_latents,
    build_time_grid,
)
from vllm_omni.diffusion.models.auk.batching import AuKBatchedStepRunner, row_bucket

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

LATENT_DIM = 4
TEXT_DIM = 8


def _make_dit(attn_mask_enabled: bool = True) -> AuKTransformer:
    torch.manual_seed(3)
    return AuKTransformer(
        dim=32,
        heads=2,
        dim_head=16,
        ff_mult=2,
        latent_dim=LATENT_DIM,
        text_hidden_dim=TEXT_DIM,
        num_layers=2,
        num_single_layers=2,
        attn_mask_enabled=attn_mask_enabled,
    ).eval()


def _request_inputs(seed: int, text_len: int, ref_len: int, frames: int, nfe: int):
    generator = torch.Generator().manual_seed(seed)
    text = torch.randn(1, text_len, TEXT_DIM, generator=generator)
    ref = torch.randn(1, ref_len, LATENT_DIM, generator=generator)
    latents = torch.randn(1, frames, LATENT_DIM, generator=generator)
    grid = build_time_grid(nfe=nfe, sway_sampling_coef=-1.0, t_grid=None, device="cpu")
    return text, ref, latents, grid


# (seed, text tokens, reference frames, target frames, steps, cfg)
_REQUESTS = [
    (0, 7, 0, 9, 4, 2.0),
    (1, 5, 6, 13, 6, 0.0),
    (2, 11, 4, 5, 3, 3.0),
]


def _single_velocity(dit: AuKTransformer, text, ref, x, grid, cfg: float, step: int) -> torch.Tensor:
    guided = cfg >= 1e-5
    ctx = dit.prepare(
        text,
        target_len=x.shape[1],
        c_mask=torch.ones(text.shape[:2], dtype=torch.bool),
        ref=ref,
        ref_mask=torch.ones(ref.shape[:2], dtype=torch.bool),
        cfg_infer=guided,
        timesteps=grid[:-1],
    )
    velocity = dit.step(x, grid[step], ctx, step_index=step)
    if guided:
        cond, uncond = velocity.chunk(2, dim=0)
        velocity = cond + (cond - uncond) * cfg
    return velocity


@torch.inference_mode()
def test_row_buckets_cover_every_batch() -> None:
    assert [row_bucket(rows) for rows in (1, 2, 3, 5, 7, 9, 16, 17, 33)] == [1, 2, 4, 6, 8, 12, 16, 24, 64]


@torch.inference_mode()
def test_bucket_key_pads_every_sequence() -> None:
    dit = _make_dit()
    runner = AuKBatchedStepRunner(dit, enabled=False)
    requests = [
        runner.make_request(text=t, ref=r, cfg=cfg, grid=g, latents=x)
        for (seed, nt, nr, n, nfe, cfg) in _REQUESTS
        for t, r, x, g in [_request_inputs(seed, nt, nr, n, nfe)]
    ]
    # 2 + 1 + 2 rows; text 11 -> 32, reference 6 -> 50, target 13 -> 32.
    assert runner.bucket_key(requests) == (6, 32, 50, 32)
    assert runner.bucket_key(requests[:1]) == (2, 32, 0, 32)
    # The eager path pads only to the longest sequence in the batch.
    assert runner.batch_shape(requests) == (5, 11, 6, 13)


@torch.inference_mode()
@pytest.mark.parametrize("steps", [(0, 0, 0), (3, 1, 2)])
def test_batched_step_matches_each_request_alone(steps) -> None:
    dit = _make_dit()
    runner = AuKBatchedStepRunner(dit, enabled=False)
    requests, singles = [], []
    for (seed, nt, nr, n, nfe, cfg), step in zip(_REQUESTS, steps):
        text, ref, x, grid = _request_inputs(seed, nt, nr, n, nfe)
        request = runner.make_request(text=text, ref=ref, cfg=cfg, grid=grid, latents=x)
        request.step_index = step
        requests.append(request)
        singles.append(_single_velocity(dit, text, ref, x, grid, cfg, step))

    velocities = runner.velocities(requests)

    for got, want, request in zip(velocities, singles, requests):
        assert got.shape == (1, request.target_frames, LATENT_DIM)
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)


@torch.inference_mode()
def test_requests_join_and_leave_mid_flight() -> None:
    """A continuous batch reproduces each request's own Euler integration."""
    dit = _make_dit()
    runner = AuKBatchedStepRunner(dit, enabled=False)
    inputs = [_request_inputs(seed, nt, nr, n, nfe) for (seed, nt, nr, n, nfe, _) in _REQUESTS]
    cfgs = [cfg for *_, cfg in _REQUESTS]
    want = [
        _sample_latents(
            dit,
            initial_latents=x,
            text=text,
            c_mask=torch.ones(text.shape[:2], dtype=torch.bool),
            ref=ref,
            ref_mask=torch.ones(ref.shape[:2], dtype=torch.bool),
            timesteps=grid,
            cfg_strength=cfg,
        )
        for (text, ref, x, grid), cfg in zip(inputs, cfgs)
    ]

    # Request i is admitted at wave arrivals[i]; each wave advances every running request.
    arrivals = [0, 2, 3]
    pending = list(range(len(inputs)))
    running: list[tuple[int, object]] = []
    finished: dict[int, torch.Tensor] = {}
    wave = 0
    while pending or running:
        for i in [i for i in pending if arrivals[i] <= wave]:
            text, ref, x, grid = inputs[i]
            running.append((i, runner.make_request(text=text, ref=ref, cfg=cfgs[i], grid=grid, latents=x)))
            pending.remove(i)
        batch = [request for _, request in running]
        for request, velocity in zip(batch, runner.velocities(batch)):
            request.advance(velocity)
        for i, request in list(running):
            if request.done:
                finished[i] = request.latents
                running.remove((i, request))
        wave += 1

    for i, expected in enumerate(want):
        torch.testing.assert_close(finished[i], expected, rtol=1e-4, atol=1e-5)


@torch.inference_mode()
def test_batching_requires_masked_attention() -> None:
    dit = _make_dit(attn_mask_enabled=False)
    runner = AuKBatchedStepRunner(dit, enabled=False)
    text, ref, x, grid = _request_inputs(0, 5, 0, 6, 2)
    request = runner.make_request(text=text, ref=ref, cfg=2.0, grid=grid, latents=x)
    with pytest.raises(ValueError, match="attn_mask_enabled"):
        runner.velocities([request])


@pytest.mark.cuda
@torch.inference_mode()
def test_graph_replay_matches_eager_across_memberships() -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    device = torch.device("cuda")
    dit = _make_dit().to(device)
    eager = AuKBatchedStepRunner(dit, enabled=False)
    graphed = AuKBatchedStepRunner(dit, enabled=True)

    def _requests(runner, members):
        out = []
        for i in members:
            seed, nt, nr, n, nfe, cfg = _REQUESTS[i]
            text, ref, x, grid = (t.to(device) for t in _request_inputs(seed, nt, nr, n, nfe))
            request = runner.make_request(text=text, ref=ref, cfg=cfg, grid=grid, latents=x)
            request.step_index = 1
            out.append(request)
        return out

    # Two memberships with the same bucket key share one graph; the second reloads its context.
    for members in ([0, 2], [2, 0], [0, 1, 2]):
        want = eager.velocities(_requests(eager, members))
        got = graphed.velocities(_requests(graphed, members))
        for g, w in zip(got, want):
            torch.testing.assert_close(g, w, rtol=1e-4, atol=1e-5)
    assert len(graphed._cache) == 2
