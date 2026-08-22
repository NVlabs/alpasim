# SimScale DiffusionDrive Vendoring Notice

This inference-only implementation is derived from the Apache-2.0 SimScale
DiffusionDrive sources stored under `reference/SimScale`. The reference copy is
not a standalone Git checkout, so provenance is pinned by SHA-256:

| Source file | SHA-256 |
| --- | --- |
| `transfuser_config.py` | `ba22a307ce8cc7ee6e2e9333661f085c503d541cb75ddf43642cf759d3439469` |
| `transfuser_backbone.py` | `8497f3a44dd805cd269d002b2e42280d1b495460bb212f01e9fcaea89d5035a2` |
| `transfuser_model_v2.py` | `793c8590375d4c3268907a39e653f3ca9ac98fc9b636967f65d6e599d780cde5` |
| `modules/blocks.py` | `de879c63cbfaf8fe96437cb84f4a9c28392625ec6706132032afceed4ffc2776` |
| `modules/conditional_unet1d.py` | `0d6d589fb29d6e076b77bddbb3d7ab6b86c216c29e13391fb532d661a13f5cf7` |

The local DDIM subset matches the operations used from Diffusers 0.35.1.
NuPlan and NAVSIM feature builders, training code and losses, Diffusers,
unused UNet code, and unrelated dependencies have been removed. Remaining
files are modified for local imports, offline initialization, strict assets,
and inference-only execution.
