# AutoKernel -- Aggregate Optimization Report

Generated: 2026-03-21 13:04:29 UTC

## Per-Kernel Summary

| Rank | Kernel | Op Type | Status | Primary Metric | Baseline | Best | Speedup | Experiments | Kept | Keep Rate | Time (min) |
|------|--------|---------|--------|----------------|----------|------|---------|-------------|------|-----------|------------|
| 1 | kernel_layout_transform_1.py | layout_transform | DONE | recognizer_latency_ms | 14.83 ms | 14.03 ms | 1.06x | 2 | 2 | 100% | 0 |
| 2 | kernel_batchnorm_2.py | batchnorm | DONE | recognizer_latency_ms | 14.19 ms | 14.19 ms | 1.00x | 1 | 0 | 0% | 18 |
| 3 | kernel_conv2d_3.py | conv2d | DONE | recognizer_latency_ms | 14.30 ms | 14.30 ms | 1.00x | 7 | 1 | 14% | 23 |
| 4 | kernel_layout_transform_4.py | layout_transform | DONE | recognizer_latency_ms | -- | -- | 1.00x | 0 | 0 | -- | 0 |
| 6 | kernel_batchnorm_6.py | batchnorm | DONE | recognizer_latency_ms | 14.42 ms | 14.42 ms | 1.00x | 2 | 1 | 50% | 2 |
| 10 | kernel_conv2d_10.py | conv2d | DONE | recognizer_latency_ms | 14.05 ms | 14.05 ms | 1.00x | 3 | 1 | 33% | 0 |
| 12 | kernel_conv2d_12.py | conv2d | DONE | recognizer_latency_ms | 14.21 ms | 14.21 ms | 1.00x | 3 | 1 | 33% | 0 |
| 16 | kernel_conv2d_16.py | conv2d | DONE | recognizer_latency_ms | 14.10 ms | 14.10 ms | 1.00x | 4 | 1 | 25% | 10 |
| 17 | kernel_conv2d_17.py | conv2d | DONE | recognizer_latency_ms | 14.19 ms | 14.19 ms | 1.00x | 3 | 1 | 33% | 6 |
| 19 | kernel_conv2d_19.py | conv2d | DONE | recognizer_latency_ms | 14.20 ms | 14.20 ms | 1.00x | 6 | 1 | 17% | 21 |
| 20 | kernel_conv2d_20.py | conv2d | DONE | recognizer_latency_ms | 14.30 ms | 14.30 ms | 1.00x | 6 | 1 | 17% | 51 |
| 23 | kernel_conv2d_23.py | conv2d | OPTIMIZING | recognizer_latency_ms | 14.31 ms | 13.83 ms | 1.03x | 8 | 3 | 38% | 88 |
| 24 | kernel_conv2d_24.py | conv2d | PENDING | recognizer_latency_ms | -- | -- | -- | 0 | 0 | -- | 0 |
| 25 | kernel_batchnorm_25.py | batchnorm | PENDING | recognizer_latency_ms | -- | -- | -- | 0 | 0 | -- | 0 |
| 26 | kernel_conv2d_26.py | conv2d | PENDING | recognizer_latency_ms | -- | -- | -- | 0 | 0 | -- | 0 |
| 28 | kernel_conv2d_28.py | conv2d | PENDING | recognizer_latency_ms | -- | -- | -- | 0 | 0 | -- | 0 |
| 29 | kernel_conv2d_29.py | conv2d | PENDING | recognizer_latency_ms | -- | -- | -- | 0 | 0 | -- | 0 |
| 30 | kernel_conv2d_30.py | conv2d | PENDING | recognizer_latency_ms | -- | -- | -- | 0 | 0 | -- | 0 |
| 33 | kernel_conv2d_33.py | conv2d | PENDING | recognizer_latency_ms | -- | -- | -- | 0 | 0 | -- | 0 |
| 34 | kernel_matmul_34.py | matmul | PENDING | recognizer_latency_ms | -- | -- | -- | 0 | 0 | -- | 0 |
| 39 | kernel_batchnorm_39.py | batchnorm | PENDING | recognizer_latency_ms | -- | -- | -- | 0 | 0 | -- | 0 |
| 43 | kernel_batchnorm_43.py | batchnorm | PENDING | recognizer_latency_ms | -- | -- | -- | 0 | 0 | -- | 0 |
| 44 | kernel_batchnorm_44.py | batchnorm | PENDING | recognizer_latency_ms | -- | -- | -- | 0 | 0 | -- | 0 |
| 45 | kernel_conv2d_45.py | conv2d | PENDING | recognizer_latency_ms | -- | -- | -- | 0 | 0 | -- | 0 |
| 46 | kernel_conv2d_46.py | conv2d | PENDING | recognizer_latency_ms | -- | -- | -- | 0 | 0 | -- | 0 |
| 47 | kernel_batchnorm_47.py | batchnorm | PENDING | recognizer_latency_ms | -- | -- | -- | 0 | 0 | -- | 0 |
| 48 | kernel_conv2d_48.py | conv2d | PENDING | recognizer_latency_ms | -- | -- | -- | 0 | 0 | -- | 0 |
| 49 | kernel_conv2d_49.py | conv2d | PENDING | recognizer_latency_ms | -- | -- | -- | 0 | 0 | -- | 0 |
| 51 | kernel_conv2d_51.py | conv2d | PENDING | recognizer_latency_ms | -- | -- | -- | 0 | 0 | -- | 0 |
| 54 | kernel_layernorm_54.py | layernorm | PENDING | recognizer_latency_ms | -- | -- | -- | 0 | 0 | -- | 0 |
| 56 | kernel_conv2d_56.py | conv2d | PENDING | recognizer_latency_ms | -- | -- | -- | 0 | 0 | -- | 0 |
| 58 | kernel_batchnorm_58.py | batchnorm | PENDING | recognizer_latency_ms | -- | -- | -- | 0 | 0 | -- | 0 |
| 59 | kernel_conv2d_59.py | conv2d | PENDING | recognizer_latency_ms | -- | -- | -- | 0 | 0 | -- | 0 |
| 63 | kernel_conv2d_63.py | conv2d | PENDING | recognizer_latency_ms | -- | -- | -- | 0 | 0 | -- | 0 |
| 64 | kernel_conv2d_64.py | conv2d | PENDING | recognizer_latency_ms | -- | -- | -- | 0 | 0 | -- | 0 |

## Aggregate Model Speedup (Amdahl's Law)

**Estimated end-to-end model speedup: 1.00x**

Breakdown by kernel (fraction of total GPU time):

- **kernel_layout_transform_1.py**: 6.7% of GPU time, 1.06x speedup (0.4% time saved)
- **kernel_batchnorm_2.py**: 5.3% of GPU time, 1.00x speedup (0.0% time saved)
- **kernel_conv2d_3.py**: 5.3% of GPU time, 1.00x speedup (0.0% time saved)
- **kernel_layout_transform_4.py**: 4.4% of GPU time, 1.00x speedup (0.0% time saved)
- **kernel_batchnorm_6.py**: 3.3% of GPU time, 1.00x speedup (0.0% time saved)
- **kernel_conv2d_10.py**: 2.6% of GPU time, 1.00x speedup (0.0% time saved)
- **kernel_conv2d_12.py**: 2.4% of GPU time, 1.00x speedup (0.0% time saved)
- **kernel_conv2d_16.py**: 2.1% of GPU time, 1.00x speedup (0.0% time saved)
- **kernel_conv2d_17.py**: 2.1% of GPU time, 1.00x speedup (0.0% time saved)
- **kernel_conv2d_19.py**: 2.0% of GPU time, 1.00x speedup (0.0% time saved)
- **kernel_conv2d_20.py**: 1.9% of GPU time, 1.00x speedup (0.0% time saved)
- **kernel_conv2d_23.py**: 1.3% of GPU time, 1.03x speedup (0.0% time saved)
- **kernel_conv2d_24.py**: 1.3% of GPU time, 1.00x speedup (0.0% time saved)
- **kernel_batchnorm_25.py**: 1.1% of GPU time, 1.00x speedup (0.0% time saved)
- **kernel_conv2d_26.py**: 1.1% of GPU time, 1.00x speedup (0.0% time saved)
- **kernel_conv2d_28.py**: 0.8% of GPU time, 1.00x speedup (0.0% time saved)
- **kernel_conv2d_29.py**: 0.8% of GPU time, 1.00x speedup (0.0% time saved)
- **kernel_conv2d_30.py**: 0.8% of GPU time, 1.00x speedup (0.0% time saved)
- **kernel_conv2d_33.py**: 0.7% of GPU time, 1.00x speedup (0.0% time saved)
- **kernel_matmul_34.py**: 0.7% of GPU time, 1.00x speedup (0.0% time saved)
- **kernel_batchnorm_39.py**: 0.6% of GPU time, 1.00x speedup (0.0% time saved)
- **kernel_batchnorm_43.py**: 0.5% of GPU time, 1.00x speedup (0.0% time saved)
- **kernel_batchnorm_44.py**: 0.5% of GPU time, 1.00x speedup (0.0% time saved)
- **kernel_conv2d_45.py**: 0.5% of GPU time, 1.00x speedup (0.0% time saved)
- **kernel_conv2d_46.py**: 0.5% of GPU time, 1.00x speedup (0.0% time saved)
- **kernel_batchnorm_47.py**: 0.5% of GPU time, 1.00x speedup (0.0% time saved)
- **kernel_conv2d_48.py**: 0.4% of GPU time, 1.00x speedup (0.0% time saved)
- **kernel_conv2d_49.py**: 0.4% of GPU time, 1.00x speedup (0.0% time saved)
- **kernel_conv2d_51.py**: 0.4% of GPU time, 1.00x speedup (0.0% time saved)
- **kernel_layernorm_54.py**: 0.4% of GPU time, 1.00x speedup (0.0% time saved)
- **kernel_conv2d_56.py**: 0.4% of GPU time, 1.00x speedup (0.0% time saved)
- **kernel_batchnorm_58.py**: 0.3% of GPU time, 1.00x speedup (0.0% time saved)
- **kernel_conv2d_59.py**: 0.3% of GPU time, 1.00x speedup (0.0% time saved)
- **kernel_conv2d_63.py**: 0.3% of GPU time, 1.00x speedup (0.0% time saved)
- **kernel_conv2d_64.py**: 0.3% of GPU time, 1.00x speedup (0.0% time saved)

## Time Allocation

Total optimization time: 219 minutes (3.6 hours)

- kernel_layout_transform_1.py: 0 min (0%)
- kernel_batchnorm_2.py: 18 min (8%)
- kernel_conv2d_3.py: 23 min (11%)
- kernel_layout_transform_4.py: 0 min (0%)
- kernel_batchnorm_6.py: 2 min (1%)
- kernel_conv2d_10.py: 0 min (0%)
- kernel_conv2d_12.py: 0 min (0%)
- kernel_conv2d_16.py: 10 min (5%)
- kernel_conv2d_17.py: 6 min (3%)
- kernel_conv2d_19.py: 21 min (10%)
- kernel_conv2d_20.py: 51 min (23%)
- kernel_conv2d_23.py: 88 min (40%)
- kernel_conv2d_24.py: 0 min (0%)
- kernel_batchnorm_25.py: 0 min (0%)
- kernel_conv2d_26.py: 0 min (0%)
- kernel_conv2d_28.py: 0 min (0%)
- kernel_conv2d_29.py: 0 min (0%)
- kernel_conv2d_30.py: 0 min (0%)
- kernel_conv2d_33.py: 0 min (0%)
- kernel_matmul_34.py: 0 min (0%)
- kernel_batchnorm_39.py: 0 min (0%)
- kernel_batchnorm_43.py: 0 min (0%)
- kernel_batchnorm_44.py: 0 min (0%)
- kernel_conv2d_45.py: 0 min (0%)
- kernel_conv2d_46.py: 0 min (0%)
- kernel_batchnorm_47.py: 0 min (0%)
- kernel_conv2d_48.py: 0 min (0%)
- kernel_conv2d_49.py: 0 min (0%)
- kernel_conv2d_51.py: 0 min (0%)
- kernel_layernorm_54.py: 0 min (0%)
- kernel_conv2d_56.py: 0 min (0%)
- kernel_batchnorm_58.py: 0 min (0%)
- kernel_conv2d_59.py: 0 min (0%)
- kernel_conv2d_63.py: 0 min (0%)
- kernel_conv2d_64.py: 0 min (0%)

## Keep Rates

- kernel_layout_transform_1.py: 2/2 (100%)
- kernel_batchnorm_2.py: 0/1 (0%)
- kernel_conv2d_3.py: 1/7 (14%)
- kernel_batchnorm_6.py: 1/2 (50%)
- kernel_conv2d_10.py: 1/3 (33%)
- kernel_conv2d_12.py: 1/3 (33%)
- kernel_conv2d_16.py: 1/4 (25%)
- kernel_conv2d_17.py: 1/3 (33%)
- kernel_conv2d_19.py: 1/6 (17%)
- kernel_conv2d_20.py: 1/6 (17%)
- kernel_conv2d_23.py: 3/8 (38%)

## Headroom Analysis

Kernels that may still have optimization potential:

- **kernel_layout_transform_1.py** (rank 1): speedup only 1.06x (target: 2.0x)
- **kernel_batchnorm_2.py** (rank 2): speedup only 1.00x (target: 2.0x)
- **kernel_conv2d_3.py** (rank 3): speedup only 1.00x (target: 2.0x)
- **kernel_layout_transform_4.py** (rank 4): speedup only 1.00x (target: 2.0x)
- **kernel_batchnorm_6.py** (rank 6): speedup only 1.00x (target: 2.0x)
- **kernel_conv2d_10.py** (rank 10): speedup only 1.00x (target: 2.0x)
- **kernel_conv2d_12.py** (rank 12): speedup only 1.00x (target: 2.0x)
- **kernel_conv2d_16.py** (rank 16): speedup only 1.00x (target: 2.0x)
- **kernel_conv2d_17.py** (rank 17): speedup only 1.00x (target: 2.0x)
- **kernel_conv2d_19.py** (rank 19): speedup only 1.00x (target: 2.0x)
- **kernel_conv2d_20.py** (rank 20): speedup only 1.00x (target: 2.0x)
- **kernel_conv2d_23.py** (rank 23): speedup only 1.03x (target: 2.0x)
- **kernel_conv2d_24.py** (rank 24): not yet optimized
- **kernel_batchnorm_25.py** (rank 25): not yet optimized
- **kernel_conv2d_26.py** (rank 26): not yet optimized
- **kernel_conv2d_28.py** (rank 28): not yet optimized
- **kernel_conv2d_29.py** (rank 29): not yet optimized
- **kernel_conv2d_30.py** (rank 30): not yet optimized
- **kernel_conv2d_33.py** (rank 33): not yet optimized
- **kernel_matmul_34.py** (rank 34): not yet optimized
- **kernel_batchnorm_39.py** (rank 39): not yet optimized
- **kernel_batchnorm_43.py** (rank 43): not yet optimized
- **kernel_batchnorm_44.py** (rank 44): not yet optimized
- **kernel_conv2d_45.py** (rank 45): not yet optimized
- **kernel_conv2d_46.py** (rank 46): not yet optimized
- **kernel_batchnorm_47.py** (rank 47): not yet optimized
- **kernel_conv2d_48.py** (rank 48): not yet optimized
- **kernel_conv2d_49.py** (rank 49): not yet optimized
- **kernel_conv2d_51.py** (rank 51): not yet optimized
- **kernel_layernorm_54.py** (rank 54): not yet optimized
- **kernel_conv2d_56.py** (rank 56): not yet optimized
- **kernel_batchnorm_58.py** (rank 58): not yet optimized
- **kernel_conv2d_59.py** (rank 59): not yet optimized
- **kernel_conv2d_63.py** (rank 63): not yet optimized
- **kernel_conv2d_64.py** (rank 64): not yet optimized
