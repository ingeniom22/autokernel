# AutoKernel -- Aggregate Optimization Report

Generated: 2026-03-21 06:24:22 UTC

## Per-Kernel Summary

| Rank | Kernel | Op Type | Status | Primary Metric | Baseline | Best | Speedup | Experiments | Kept | Keep Rate | Time (min) |
|------|--------|---------|--------|----------------|----------|------|---------|-------------|------|-----------|------------|
| 3 | kernel_conv2d_3.py | conv2d | DONE | recognizer_latency_ms | 14.83 ms | 14.83 ms | 1.00x | 6 | 1 | 17% | 9 |
| 6 | kernel_batchnorm_6.py | batchnorm | DONE | recognizer_latency_ms | 14.53 ms | 14.53 ms | 1.00x | 6 | 1 | 17% | 17 |
| 10 | kernel_conv2d_10.py | conv2d | DONE | recognizer_latency_ms | 15.10 ms | 15.10 ms | 1.00x | 6 | 0 | 0% | 33 |
| 12 | kernel_conv2d_12.py | conv2d | OPTIMIZING | recognizer_latency_ms | -- | -- | -- | 0 | 0 | -- | 0 |
| 16 | kernel_conv2d_16.py | conv2d | PENDING | recognizer_latency_ms | -- | -- | -- | 0 | 0 | -- | 0 |
| 17 | kernel_conv2d_17.py | conv2d | PENDING | recognizer_latency_ms | -- | -- | -- | 0 | 0 | -- | 0 |
| 19 | kernel_conv2d_19.py | conv2d | PENDING | recognizer_latency_ms | -- | -- | -- | 0 | 0 | -- | 0 |
| 20 | kernel_conv2d_20.py | conv2d | PENDING | recognizer_latency_ms | -- | -- | -- | 0 | 0 | -- | 0 |
| 23 | kernel_conv2d_23.py | conv2d | PENDING | recognizer_latency_ms | -- | -- | -- | 0 | 0 | -- | 0 |
| 24 | kernel_conv2d_24.py | conv2d | PENDING | recognizer_latency_ms | -- | -- | -- | 0 | 0 | -- | 0 |
| 25 | kernel_batchnorm_25.py | batchnorm | PENDING | recognizer_latency_ms | -- | -- | -- | 0 | 0 | -- | 0 |
| 26 | kernel_conv2d_26.py | conv2d | PENDING | recognizer_latency_ms | -- | -- | -- | 0 | 0 | -- | 0 |

## Aggregate Model Speedup (Amdahl's Law)

No measurable aggregate speedup yet.

Breakdown by kernel (fraction of total GPU time):

- **kernel_conv2d_3.py**: 5.3% of GPU time, 1.00x speedup (0.0% time saved)
- **kernel_batchnorm_6.py**: 3.3% of GPU time, 1.00x speedup (0.0% time saved)
- **kernel_conv2d_10.py**: 2.6% of GPU time, 1.00x speedup (0.0% time saved)
- **kernel_conv2d_12.py**: 2.4% of GPU time, 1.00x speedup (0.0% time saved)
- **kernel_conv2d_16.py**: 2.1% of GPU time, 1.00x speedup (0.0% time saved)
- **kernel_conv2d_17.py**: 2.1% of GPU time, 1.00x speedup (0.0% time saved)
- **kernel_conv2d_19.py**: 2.0% of GPU time, 1.00x speedup (0.0% time saved)
- **kernel_conv2d_20.py**: 1.9% of GPU time, 1.00x speedup (0.0% time saved)
- **kernel_conv2d_23.py**: 1.3% of GPU time, 1.00x speedup (0.0% time saved)
- **kernel_conv2d_24.py**: 1.3% of GPU time, 1.00x speedup (0.0% time saved)
- **kernel_batchnorm_25.py**: 1.1% of GPU time, 1.00x speedup (0.0% time saved)
- **kernel_conv2d_26.py**: 1.1% of GPU time, 1.00x speedup (0.0% time saved)

## Time Allocation

Total optimization time: 59 minutes (1.0 hours)

- kernel_conv2d_3.py: 9 min (15%)
- kernel_batchnorm_6.py: 17 min (29%)
- kernel_conv2d_10.py: 33 min (56%)
- kernel_conv2d_12.py: 0 min (0%)
- kernel_conv2d_16.py: 0 min (0%)
- kernel_conv2d_17.py: 0 min (0%)
- kernel_conv2d_19.py: 0 min (0%)
- kernel_conv2d_20.py: 0 min (0%)
- kernel_conv2d_23.py: 0 min (0%)
- kernel_conv2d_24.py: 0 min (0%)
- kernel_batchnorm_25.py: 0 min (0%)
- kernel_conv2d_26.py: 0 min (0%)

## Keep Rates

- kernel_conv2d_3.py: 1/6 (17%)
- kernel_batchnorm_6.py: 1/6 (17%)
- kernel_conv2d_10.py: 0/6 (0%)

## Headroom Analysis

Kernels that may still have optimization potential:

- **kernel_conv2d_3.py** (rank 3): speedup only 1.00x (target: 2.0x)
- **kernel_batchnorm_6.py** (rank 6): speedup only 1.00x (target: 2.0x)
- **kernel_conv2d_10.py** (rank 10): speedup only 1.00x (target: 2.0x)
- **kernel_conv2d_16.py** (rank 16): not yet optimized
- **kernel_conv2d_17.py** (rank 17): not yet optimized
- **kernel_conv2d_19.py** (rank 19): not yet optimized
- **kernel_conv2d_20.py** (rank 20): not yet optimized
- **kernel_conv2d_23.py** (rank 23): not yet optimized
- **kernel_conv2d_24.py** (rank 24): not yet optimized
- **kernel_batchnorm_25.py** (rank 25): not yet optimized
- **kernel_conv2d_26.py** (rank 26): not yet optimized
