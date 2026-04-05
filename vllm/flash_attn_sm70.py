"""Compatibility shim for the SM70 Flash Attention backend.

This is the package-local entrypoint used by `from vllm import flash_attn_sm70`.
It re-exports the SM70 FlashAttention-2 varlen implementation and keeps the
paged decode hook optional so the backend can fall back to Triton cleanly.
"""

from vllm.vllm_flash_attn_sm70.flash_attn_interface import (
    flash_attn_varlen_func,
)

# The current SM70 port only provides the dense varlen path.
# Decode can remain on Triton until a paged KV implementation is added.
flash_attn_func = flash_attn_varlen_func
flash_attn_decode_paged = None

__all__ = [
    "flash_attn_varlen_func",
    "flash_attn_func",
    "flash_attn_decode_paged",
]
