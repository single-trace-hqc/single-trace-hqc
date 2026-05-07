# Single-Trace Key Recovery Attacks on HQC Using Valid and Invalid Ciphertexts

This is the repository for the paper **Single-Trace Key Recovery Attacks on HQC Using Valid and Invalid Ciphertexts** ([ePrint 2025/1987](https://eprint.iacr.org/2025/1987), to appear in Eurocrypt 2026). 

As in the paper, the valid-ciphertext attack is referred to as VCMDPC, and the invalid-ciphertext attack is referred to as OTFDACK.

Repository structure:

```text
single-trace-hqc/
├── hqc-v5.0.0/
│   └── HQC reference implementation used by the scripts
└── src/
    ├── Makefile                    # build MDPC bp decoder
    ├── calc_num_errors_OTFDACK.py  # statistics for OTFDACK (Fig. 4)
    ├── calc_num_errors_VCMDPC.py   # statistics for VCMDPC (Fig. 3)
    ├── config.json                 # parameters and paths for HQC schemes
    ├── create_template.py          # create OTs for OTFDACK
    ├── find_error_patterns.py      # optimize error patterns for OTFDACK
    ├── isd.py                      # ISD simulation helper
    ├── mdpcbp.c                    # MDPC belief-propagation decoder for VCMDPC
    ├── simulation_OTFDACK.py       # ISD simulation for OTFDACK
    ├── simulation_VCMDPC.py        # ISD simulation for VCMDPC
    └── util.py                     # utility functions
```
