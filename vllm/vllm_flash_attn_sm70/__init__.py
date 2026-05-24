__version__ = "2.7.2.post1"

# Use relative import to support build-from-source installation in vLLM
from .flash_attn_interface import (
    fa_version_unsupported_reason,
    flash_attn_decode_paged,
    flash_attn_decode_partitioned,
    flash_attn_prefill_paged,
    flash_attn_varlen_func,
    get_scheduler_metadata,
    is_fa_version_supported,
)

# Keep the package-level API aligned with the SM70 shim used by the backend.
flash_attn_func = flash_attn_varlen_func

__all__ = [
    "fa_version_unsupported_reason",
    "flash_attn_decode_paged",
    "flash_attn_decode_partitioned",
    "flash_attn_prefill_paged",
    "flash_attn_varlen_func",
    "flash_attn_func",
    "get_scheduler_metadata",
    "is_fa_version_supported",
]
