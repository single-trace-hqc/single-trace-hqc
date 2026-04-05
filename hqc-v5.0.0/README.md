# HQC Ref Build Notes

This folder is for building the HQC reference implementation.
Upstream HQC is used as a Git submodule. 
`Makefile` builds `hqc-1` by default. 
It builds a shared library.

Submodule path:
`hqc-v5.0.0/next-release`

Source URL:
`https://gitlab.com/pqc-hqc/hqc.git`

Tracked branch:
`next-release`

Pinned commit:
`f46e542`

## Patches

Two local patches are applied during the build in `build/ref/patched/next-release`.
The original files in `next-release` are not modified by this build flow.

1. `patches/decodeOneRMBlock.patch`
Adds `reed_muller_decode_one_block(...)`, which returns the RM decode result for one block, plus the header declaration.

2. `patches/ref_arm64_no_immintrin.patch`
Wraps `<immintrin.h>` includes with x86 checks, so ref build also works on arm64.