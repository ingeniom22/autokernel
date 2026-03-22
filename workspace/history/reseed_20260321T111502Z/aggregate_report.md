# AutoKernel -- Aggregate Optimization Report

Generated: 2026-03-21 11:00:13 UTC

## Per-Kernel Summary

| Rank | Kernel | Op Type | Status | Primary Metric | Baseline | Best | Speedup | Experiments | Kept | Keep Rate | Time (min) |
|------|--------|---------|--------|----------------|----------|------|---------|-------------|------|-----------|------------|
| 1 | kernel_layout_transform_1.py | layout_transform | DONE | recognizer_latency_ms | 14.83 ms | 14.03 ms | 1.06x | 2 | 2 | 100% | 0 |
| 2 | kernel_batchnorm_2.py | batchnorm | DONE | recognizer_latency_ms | 14.19 ms | 14.19 ms | 1.00x | 1 | 0 | 0% | 18 |
| 3 | kernel_conv2d_3.py | conv2d | DONE | recognizer_latency_ms | 14.30 ms | 14.30 ms | 1.00x | 1 | 0 | 0% | 23 |
| 4 | kernel_layout_transform_4.py | layout_transform | DONE | recognizer_latency_ms | -- | -- | 1.00x | 0 | 0 | -- | 0 |
| 5 | kernel_matmul_5.py | matmul | DONE | recognizer_latency_ms | 14.19 ms | 14.19 ms | 1.00x | 2 | 1 | 50% | 37 |
| 6 | kernel_batchnorm_6.py | batchnorm | DONE | recognizer_latency_ms | 14.42 ms | 14.42 ms | 1.00x | 2 | 1 | 50% | 2 |
| 7 | kernel_batchnorm_7.py | batchnorm | DONE | recognizer_latency_ms | 13.83 ms | 13.83 ms | 1.00x | 2 | 1 | 50% | 2 |
| 8 | kernel_matmul_8.py | matmul | SKIPPED | recognizer_latency_ms | -- | -- | -- | 0 | 0 | -- | 0 |
| 9 | kernel_conv2d_9.py | conv2d | SKIPPED | recognizer_latency_ms | -- | -- | -- | 0 | 0 | -- | 0 |
| 10 | kernel_conv2d_10.py | conv2d | DONE | recognizer_latency_ms | 14.05 ms | 14.05 ms | 1.00x | 3 | 1 | 33% | 0 |
| 11 | kernel_matmul_11.py | matmul | SKIPPED | recognizer_latency_ms | -- | -- | -- | 0 | 0 | -- | 0 |
| 12 | kernel_conv2d_12.py | conv2d | DONE | recognizer_latency_ms | 14.21 ms | 14.21 ms | 1.00x | 3 | 1 | 33% | 0 |
| 70 | kernel_softmax_70.py | softmax | DONE | recognizer_latency_ms | 14.20 ms | 14.20 ms | 1.00x | 2 | 1 | 50% | 1 |
| 54 | kernel_layernorm_54.py | layernorm | DONE | recognizer_latency_ms | 14.00 ms | 14.00 ms | 1.00x | 3 | 1 | 33% | 0 |

## Aggregate Model Speedup (Amdahl's Law)

**Estimated end-to-end model speedup: 1.00x**

Breakdown by kernel (fraction of total GPU time):

- **kernel_layout_transform_1.py**: 6.7% of GPU time, 1.06x speedup (0.4% time saved)
- **kernel_batchnorm_2.py**: 5.3% of GPU time, 1.00x speedup (0.0% time saved)
- **kernel_conv2d_3.py**: 5.3% of GPU time, 1.00x speedup (0.0% time saved)
- **kernel_layout_transform_4.py**: 4.4% of GPU time, 1.00x speedup (0.0% time saved)
- **kernel_matmul_5.py**: 3.6% of GPU time, 1.00x speedup (0.0% time saved)
- **kernel_batchnorm_6.py**: 3.3% of GPU time, 1.00x speedup (0.0% time saved)
- **kernel_batchnorm_7.py**: 3.1% of GPU time, 1.00x speedup (0.0% time saved)
- **kernel_matmul_8.py**: 2.8% of GPU time, 1.00x speedup (0.0% time saved)
- **kernel_conv2d_9.py**: 2.7% of GPU time, 1.00x speedup (0.0% time saved)
- **kernel_conv2d_10.py**: 2.6% of GPU time, 1.00x speedup (0.0% time saved)
- **kernel_matmul_11.py**: 2.6% of GPU time, 1.00x speedup (0.0% time saved)
- **kernel_conv2d_12.py**: 2.4% of GPU time, 1.00x speedup (0.0% time saved)
- **kernel_softmax_70.py**: 0.2% of GPU time, 1.00x speedup (0.0% time saved)
- **kernel_layernorm_54.py**: 0.4% of GPU time, 1.00x speedup (0.0% time saved)

## Time Allocation

Total optimization time: 83 minutes (1.4 hours)

- kernel_layout_transform_1.py: 0 min (0%)
- kernel_batchnorm_2.py: 18 min (22%)
- kernel_conv2d_3.py: 23 min (28%)
- kernel_layout_transform_4.py: 0 min (0%)
- kernel_matmul_5.py: 37 min (45%)
- kernel_batchnorm_6.py: 2 min (2%)
- kernel_batchnorm_7.py: 2 min (2%)
- kernel_matmul_8.py: 0 min (0%)
- kernel_conv2d_9.py: 0 min (0%)
- kernel_conv2d_10.py: 0 min (0%)
- kernel_matmul_11.py: 0 min (0%)
- kernel_conv2d_12.py: 0 min (0%)
- kernel_softmax_70.py: 1 min (1%)
- kernel_layernorm_54.py: 0 min (0%)

## Keep Rates

- kernel_layout_transform_1.py: 2/2 (100%)
- kernel_batchnorm_2.py: 0/1 (0%)
- kernel_conv2d_3.py: 0/1 (0%)
- kernel_matmul_5.py: 1/2 (50%)
- kernel_batchnorm_6.py: 1/2 (50%)
- kernel_batchnorm_7.py: 1/2 (50%)
- kernel_conv2d_10.py: 1/3 (33%)
- kernel_conv2d_12.py: 1/3 (33%)
- kernel_softmax_70.py: 1/2 (50%)
- kernel_layernorm_54.py: 1/3 (33%)

## Headroom Analysis

Kernels that may still have optimization potential:

- **kernel_layout_transform_1.py** (rank 1): speedup only 1.06x (target: 2.0x)
- **kernel_batchnorm_2.py** (rank 2): speedup only 1.00x (target: 2.0x)
- **kernel_conv2d_3.py** (rank 3): speedup only 1.00x (target: 2.0x)
- **kernel_layout_transform_4.py** (rank 4): speedup only 1.00x (target: 2.0x)
- **kernel_matmul_5.py** (rank 5): speedup only 1.00x (target: 2.0x)
- **kernel_batchnorm_6.py** (rank 6): speedup only 1.00x (target: 2.0x)
- **kernel_batchnorm_7.py** (rank 7): speedup only 1.00x (target: 2.0x)
- **kernel_conv2d_10.py** (rank 10): speedup only 1.00x (target: 2.0x)
- **kernel_conv2d_12.py** (rank 12): speedup only 1.00x (target: 2.0x)
- **kernel_softmax_70.py** (rank 70): speedup only 1.00x (target: 2.0x)
- **kernel_layernorm_54.py** (rank 54): speedup only 1.00x (target: 2.0x)
