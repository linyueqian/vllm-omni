"""Unit test for the .item()-in-inner-loop fix in _LocalPredictorKVCache.build_attn_metadata.

Tests that the fixed slot_mapping computation (using batch .tolist() instead
of per-element .item()) produces identical results to the original code.
Runs on CPU only — no GPU or model weights required.
"""

import torch


def build_slot_mapping_original(
    num_reqs: int,
    query_lens_i32: torch.Tensor,
    seq_lens_i32: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Original code from lines 124-147 (pre-fix): .item() in every loop iteration."""
    qsl = torch.zeros((num_reqs + 1,), dtype=torch.int32, device="cpu")
    qsl[1:] = torch.cumsum(query_lens_i32[:num_reqs], dim=0)
    num_tokens = int(qsl[-1].item())

    pos_list = []
    for i in range(num_reqs):
        ql = int(query_lens_i32[i].item())
        sl = int(seq_lens_i32[i].item())
        start = sl - ql
        pos_list.append(torch.arange(start, sl, dtype=torch.int64))
    positions_cpu = torch.cat(pos_list, dim=0)

    slot_mapping = torch.empty((num_tokens,), dtype=torch.int64, device="cpu")
    cursor = 0
    for i in range(num_reqs):
        ql = int(query_lens_i32[i].item())
        sl = int(seq_lens_i32[i].item())
        start = sl - ql
        for p in range(start, sl):
            block_idx = p // block_size
            offset = p % block_size
            block_id = int(block_table[i, block_idx].item())
            slot_mapping[cursor] = block_id * block_size + offset
            cursor += 1

    return positions_cpu, slot_mapping


def build_slot_mapping_fixed(
    num_reqs: int,
    query_lens_i32: torch.Tensor,
    seq_lens_i32: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed code: batch .tolist() conversion, no per-element .item()."""
    qsl = torch.zeros((num_reqs + 1,), dtype=torch.int32, device="cpu")
    qsl[1:] = torch.cumsum(query_lens_i32[:num_reqs], dim=0)
    num_tokens = int(qsl[-1].item())

    query_lens_list = query_lens_i32[:num_reqs].tolist()
    seq_lens_list = seq_lens_i32[:num_reqs].tolist()
    block_table_cpu = block_table[:num_reqs].cpu().tolist()

    pos_list = []
    for i in range(num_reqs):
        ql = query_lens_list[i]
        sl = seq_lens_list[i]
        start = sl - ql
        pos_list.append(torch.arange(start, sl, dtype=torch.int64))
    positions_cpu = torch.cat(pos_list, dim=0)

    slot_mapping = torch.empty((num_tokens,), dtype=torch.int64, device="cpu")
    cursor = 0
    for i in range(num_reqs):
        ql = query_lens_list[i]
        sl = seq_lens_list[i]
        start = sl - ql
        for p in range(start, sl):
            block_idx = p // block_size
            offset = p % block_size
            block_id = block_table_cpu[i][block_idx]
            slot_mapping[cursor] = block_id * block_size + offset
            cursor += 1

    return positions_cpu, slot_mapping


def make_block_table(num_reqs, blocks_per_seq):
    bt = torch.full((num_reqs, blocks_per_seq), -1, dtype=torch.int32)
    for i in range(num_reqs):
        for j in range(blocks_per_seq):
            bt[i, j] = i * blocks_per_seq + j
    return bt


def test_decode_single_request():
    """Single request, decode mode (query_len=1)."""
    block_size = 16
    num_reqs = 1
    query_lens = torch.tensor([1], dtype=torch.int32)
    seq_lens = torch.tensor([5], dtype=torch.int32)
    bt = make_block_table(num_reqs, 1)

    pos_orig, sm_orig = build_slot_mapping_original(num_reqs, query_lens, seq_lens, bt, block_size)
    pos_fixed, sm_fixed = build_slot_mapping_fixed(num_reqs, query_lens, seq_lens, bt, block_size)

    assert torch.equal(pos_orig, pos_fixed), f"positions differ: {pos_orig} vs {pos_fixed}"
    assert torch.equal(sm_orig, sm_fixed), f"slot_mapping differ: {sm_orig} vs {sm_fixed}"
    # Position should be seq_len - query_len = 4
    assert pos_orig.tolist() == [4], f"expected [4], got {pos_orig.tolist()}"
    print("PASS: test_decode_single_request")


def test_prefill_single_request():
    """Single request, prefill mode (query_len=2, typical for code predictor)."""
    block_size = 16
    num_reqs = 1
    query_lens = torch.tensor([2], dtype=torch.int32)
    seq_lens = torch.tensor([2], dtype=torch.int32)
    bt = make_block_table(num_reqs, 1)

    pos_orig, sm_orig = build_slot_mapping_original(num_reqs, query_lens, seq_lens, bt, block_size)
    pos_fixed, sm_fixed = build_slot_mapping_fixed(num_reqs, query_lens, seq_lens, bt, block_size)

    assert torch.equal(pos_orig, pos_fixed), "positions differ"
    assert torch.equal(sm_orig, sm_fixed), "slot_mapping differ"
    assert pos_orig.tolist() == [0, 1], f"expected [0,1], got {pos_orig.tolist()}"
    print("PASS: test_prefill_single_request")


def test_batch_decode():
    """Multiple requests in batch, decode mode."""
    block_size = 16
    num_reqs = 4
    query_lens = torch.ones((num_reqs,), dtype=torch.int32)
    seq_lens = torch.tensor([3, 7, 1, 12], dtype=torch.int32)
    bt = make_block_table(num_reqs, 1)

    pos_orig, sm_orig = build_slot_mapping_original(num_reqs, query_lens, seq_lens, bt, block_size)
    pos_fixed, sm_fixed = build_slot_mapping_fixed(num_reqs, query_lens, seq_lens, bt, block_size)

    assert torch.equal(pos_orig, pos_fixed), "positions differ"
    assert torch.equal(sm_orig, sm_fixed), "slot_mapping differ"
    # positions should be [2, 6, 0, 11]
    assert pos_orig.tolist() == [2, 6, 0, 11], f"expected [2,6,0,11], got {pos_orig.tolist()}"
    print("PASS: test_batch_decode")


def test_cross_block_boundary():
    """Request whose tokens span multiple blocks."""
    block_size = 4
    num_reqs = 1
    query_lens = torch.tensor([3], dtype=torch.int32)
    seq_lens = torch.tensor([6], dtype=torch.int32)  # positions 3,4,5 → blocks 0 and 1
    bt = make_block_table(num_reqs, 2)

    pos_orig, sm_orig = build_slot_mapping_original(num_reqs, query_lens, seq_lens, bt, block_size)
    pos_fixed, sm_fixed = build_slot_mapping_fixed(num_reqs, query_lens, seq_lens, bt, block_size)

    assert torch.equal(pos_orig, pos_fixed), "positions differ"
    assert torch.equal(sm_orig, sm_fixed), "slot_mapping differ"
    # pos 3 → block 0, slot 3; pos 4 → block 1, slot 4; pos 5 → block 1, slot 5
    # block_id for block_idx=0 is bt[0,0]=0, block_idx=1 is bt[0,1]=1
    # slot(3) = 0*4 + 3 = 3
    # slot(4) = 1*4 + 0 = 4
    # slot(5) = 1*4 + 1 = 5
    assert sm_orig.tolist() == [3, 4, 5], f"expected [3,4,5], got {sm_orig.tolist()}"
    print("PASS: test_cross_block_boundary")


def test_large_batch():
    """Stress test with a larger batch to catch edge cases."""
    block_size = 16
    num_reqs = 32
    query_lens = torch.ones((num_reqs,), dtype=torch.int32)
    seq_lens = torch.arange(1, num_reqs + 1, dtype=torch.int32)
    blocks_per_seq = (int(seq_lens.max()) + block_size - 1) // block_size
    bt = make_block_table(num_reqs, blocks_per_seq)

    pos_orig, sm_orig = build_slot_mapping_original(num_reqs, query_lens, seq_lens, bt, block_size)
    pos_fixed, sm_fixed = build_slot_mapping_fixed(num_reqs, query_lens, seq_lens, bt, block_size)

    assert torch.equal(pos_orig, pos_fixed), "positions differ"
    assert torch.equal(sm_orig, sm_fixed), "slot_mapping differ"
    print("PASS: test_large_batch")


def test_mixed_query_lens():
    """Requests with different query lengths."""
    block_size = 8
    num_reqs = 3
    query_lens = torch.tensor([1, 2, 1], dtype=torch.int32)
    seq_lens = torch.tensor([4, 2, 8], dtype=torch.int32)
    blocks_per_seq = (int(seq_lens.max()) + block_size - 1) // block_size
    bt = make_block_table(num_reqs, blocks_per_seq)

    pos_orig, sm_orig = build_slot_mapping_original(num_reqs, query_lens, seq_lens, bt, block_size)
    pos_fixed, sm_fixed = build_slot_mapping_fixed(num_reqs, query_lens, seq_lens, bt, block_size)

    assert torch.equal(pos_orig, pos_fixed), "positions differ"
    assert torch.equal(sm_orig, sm_fixed), "slot_mapping differ"
    # req 0: pos [3]; req 1: pos [0,1]; req 2: pos [7]
    assert pos_orig.tolist() == [3, 0, 1, 7], f"expected [3,0,1,7], got {pos_orig.tolist()}"
    print("PASS: test_mixed_query_lens")


if __name__ == "__main__":
    test_decode_single_request()
    test_prefill_single_request()
    test_batch_decode()
    test_cross_block_boundary()
    test_large_batch()
    test_mixed_query_lens()
    print("\nAll tests PASSED — original and fixed code produce identical results.")
