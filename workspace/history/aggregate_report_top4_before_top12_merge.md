# AutoKernel -- Aggregate Optimization Report

Generated: 2026-03-21 06:49:28 UTC

## Per-Kernel Summary

| Rank | Kernel | Op Type | Status | Primary Metric | Baseline | Best | Speedup | Experiments | Kept | Keep Rate | Time (min) |
|------|--------|---------|--------|----------------|----------|------|---------|-------------|------|-----------|------------|
| 1 | kernel_layout_transform_1.py | layout_transform | OPTIMIZING | recognizer_latency_ms | 14.83 ms | 14.03 ms | 1.06x | 2 | 2 | 100% | 0 |
| 2 | kernel_batchnorm_2.py | batchnorm | PENDING | recognizer_latency_ms | -- | -- | -- | 0 | 0 | -- | 0 |
| 3 | kernel_conv2d_3.py | conv2d | PENDING | recognizer_latency_ms | -- | -- | -- | 0 | 0 | -- | 0 |
| 4 | kernel_layout_transform_4.py | layout_transform | PENDING | recognizer_latency_ms | -- | -- | -- | 0 | 0 | -- | 0 |

## Aggregate Model Speedup (Amdahl's Law)

**Estimated end-to-end model speedup: 1.00x**

Breakdown by kernel (fraction of total GPU time):

- **kernel_layout_transform_1.py**: 6.7% of GPU time, 1.06x speedup (0.4% time saved)
- **kernel_batchnorm_2.py**: 5.3% of GPU time, 1.00x speedup (0.0% time saved)
- **kernel_conv2d_3.py**: 5.3% of GPU time, 1.00x speedup (0.0% time saved)
- **kernel_layout_transform_4.py**: 4.4% of GPU time, 1.00x speedup (0.0% time saved)

## Time Allocation

No time tracked yet.

## Keep Rates

- kernel_layout_transform_1.py: 2/2 (100%)

## Headroom Analysis

Kernels that may still have optimization potential:

- **kernel_layout_transform_1.py** (rank 1): speedup only 1.06x (target: 2.0x)
- **kernel_batchnorm_2.py** (rank 2): not yet optimized
- **kernel_conv2d_3.py** (rank 3): not yet optimized
- **kernel_layout_transform_4.py** (rank 4): not yet optimized
