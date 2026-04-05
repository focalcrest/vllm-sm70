# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch


def test_flash_attn_sm70_decode_wrapper_normalizes_decode_inputs(monkeypatch):
    try:
        from vllm.vllm_flash_attn_sm70.flash_attn_interface import (
            flash_attn_decode_paged,
        )
    except ImportError as exc:  # pragma: no cover - build-dependent import
        pytest.skip(f"SM70 flash-attn extension unavailable: {exc}")

    q = torch.randn(2, 3, 4)
    k_cache = torch.randn(8, 3, 4)
    v_cache = torch.randn(8, 3, 4)
    block_table = torch.arange(6, dtype=torch.int64).view(3, 2).transpose(0, 1)
    seq_lens = torch.tensor([4, 5, 6, 7], dtype=torch.int64)[::2]
    out = torch.zeros(2, 12)

    def fake_fwd_kvcache(
        q_arg,
        k_cache_arg,
        v_cache_arg,
        *_args,
    ):
        seq_lens_arg = _args[2]
        block_table_arg = _args[7]
        out_arg = _args[9]

        assert q_arg.shape == (2, 1, 3, 4)
        assert block_table_arg.dtype == torch.int32
        assert block_table_arg.is_contiguous()
        assert seq_lens_arg.dtype == torch.int32
        assert seq_lens_arg.is_contiguous()
        assert out_arg.shape == (2, 1, 3, 4)

        out_arg.fill_(1)
        return (out_arg, torch.empty(0, device=out_arg.device))

    monkeypatch.setattr(
        torch.ops,
        "_vllm_fa2_sm70_C",
        SimpleNamespace(fwd_kvcache=fake_fwd_kvcache),
        raising=False,
    )

    result = flash_attn_decode_paged(
        q,
        k_cache,
        v_cache,
        block_table,
        seq_lens,
        out=out,
    )

    assert result.shape == (2, 1, 3, 4)
    assert torch.all(result == 1)
    assert torch.all(out.view(2, 1, 3, 4) == 1)
