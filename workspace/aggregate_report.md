# AutoKernel -- Aggregate Optimization Report

Generated: 2026-03-21 05:02:12 UTC

## Per-Kernel Summary

| Rank | Kernel | Op Type | Status | Baseline (TFLOPS) | Best (TFLOPS) | Speedup | Experiments | Kept | Keep Rate | Time (min) |
|------|--------|---------|--------|-------------------|---------------|---------|-------------|------|-----------|------------|
| 34 | kernel_matmul_34.py | matmul | DONE | 0.18 | 0.18 | 1.00x | 4 | 1 | 25% | 7 |
| 54 | kernel_layernorm_54.py | layernorm | DONE | 0.99 | 1.00 | 1.00x | 7 | 4 | 57% | 43 |
| 70 | kernel_softmax_70.py | softmax | DONE | 0.64 | 0.65 | 1.01x | 7 | 6 | 86% | 44 |
| 88 | kernel_matmul_88.py | matmul | DONE | 0.18 | 0.18 | 1.00x | 2 | 1 | 50% | 29 |
| 89 | kernel_matmul_89.py | matmul | DONE | 0.18 | 0.18 | 1.00x | 2 | 1 | 50% | 33 |

## Aggregate Model Speedup (Amdahl's Law)

**Estimated end-to-end model speedup: 1.00x**

Breakdown by kernel (fraction of total GPU time):

- **kernel_matmul_34.py**: 0.7% of GPU time, 1.00x speedup (0.0% time saved)
- **kernel_layernorm_54.py**: 0.4% of GPU time, 1.00x speedup (0.0% time saved)
- **kernel_softmax_70.py**: 0.2% of GPU time, 1.01x speedup (0.0% time saved)
- **kernel_matmul_88.py**: 0.2% of GPU time, 1.00x speedup (0.0% time saved)
- **kernel_matmul_89.py**: 0.2% of GPU time, 1.00x speedup (0.0% time saved)

## Time Allocation

Total optimization time: 156 minutes (2.6 hours)

- kernel_matmul_34.py: 7 min (4%)
- kernel_layernorm_54.py: 43 min (28%)
- kernel_softmax_70.py: 44 min (28%)
- kernel_matmul_88.py: 29 min (19%)
- kernel_matmul_89.py: 33 min (21%)

## Keep Rates

- kernel_matmul_34.py: 1/4 (25%)
- kernel_layernorm_54.py: 4/7 (57%)
- kernel_softmax_70.py: 6/7 (86%)
- kernel_matmul_88.py: 1/2 (50%)
- kernel_matmul_89.py: 1/2 (50%)

## Headroom Analysis

Kernels that may still have optimization potential:

- **kernel_matmul_34.py** (rank 34): speedup only 1.00x (target: 2.0x)
- **kernel_layernorm_54.py** (rank 54): speedup only 1.00x (target: 2.0x)
- **kernel_softmax_70.py** (rank 70): speedup only 1.01x (target: 2.0x)
- **kernel_matmul_88.py** (rank 88): speedup only 1.00x (target: 2.0x)
- **kernel_matmul_89.py** (rank 89): speedup only 1.00x (target: 2.0x)
