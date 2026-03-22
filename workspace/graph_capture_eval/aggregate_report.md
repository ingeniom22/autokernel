# Graph Capture Eval Report

Generated: 2026-03-21 UTC

## Accepted Stack

- `layout_transform_1`: `14.83 ms -> 14.03 ms` (`1.057x`)
- `graph_capture_900`: `14.21 ms -> 2.03 ms` (`6.998x`)
- `conv2d_23`: `2.03 ms -> 2.01 ms` (`1.008x`), confirmed on `20/400`
- `conv2d_20_graph`: `2.01 ms -> 1.94 ms` (`1.038x`), confirmed on `20/400`
- `block_fusion_901`: `1.93 ms -> 1.84 ms` (`1.053x`), confirmed on `20/400`
- `block_fusion_902`: `1.83 ms -> 1.76 ms` (`1.039x`), confirmed on `20/400`
- `block_fusion_904`: `1.77 ms -> 1.72 ms` (`1.025x`), confirmed on `20/400`
- `block_fusion_905`: `1.73 ms -> 1.71 ms` (`1.012x`), confirmed on `20/400`
- `block_fusion_906`: `1.70 ms -> 1.68 ms` (`1.015x`), confirmed on `20/400`
- `block_fusion_908`: `1.69 ms -> 1.56 ms` (`1.082x`), Inductor-compiled stage2 wrapper replacing the earlier eager `block_fusion_901`, confirmed on `20/400`
- `block_fusion_909`: `1.54 ms -> 1.50 ms` (`1.029x`), Inductor-compiled SVTR encoder wrapper replacing the earlier eager `block_fusion_906`, confirmed on repeated `20/400` runs
- `block_fusion_910`: `1.50 ms -> 1.47 ms` (`1.024x`), Inductor-compiled stem wrapper replacing the earlier eager `block_fusion_905`, confirmed on `20/400`
- `block_fusion_911`: `1.47 ms -> 1.40 ms` (`1.051x`), Inductor-compiled original stage0 wrapper replacing the rejected folded stage0 path, confirmed on `20/400`
- `block_fusion_921`: `1.41 ms -> 1.36 ms` (`1.031x`), Inductor-compiled full stage3 wrapper preserving the accepted `block_fusion_902` block0 specialization, confirmed on `20/400`
- `block_fusion_922`: `1.36 ms -> 1.33 ms` (`1.025x`), Inductor-compiled full backbone wrapper collapsing the accepted inner specializations behind a single outer boundary, confirmed on `20/400`

Current accepted recognizer latency in this workspace: `1.33 ms` on `PPOCRv5ServerRecModel` with `--warmup 20 --timed 400`.

## Rejected Experiments

- `conv2d_10_graph`: `1.93 ms -> 1.93 ms` on the `20/400` confirmation run, so it was reverted as neutral.
- `conv2d_17_graph`: failed correctness at the active stack tolerance with `max_abs_error=3.17e-03`, so it was reverted even though the raw latency was not worse.
- `block_fusion_903`: `1.76 ms -> 1.71 ms` on the `20/200` screen, but it failed correctness at the active tolerance with `max_abs_error=4.39e-03`, so it was reverted.
- `block_fusion_907`: the SVTR block-level SDPA rewrite looked fast on a `10/100` reference-stack screen (`2.3 ms -> 1.7 ms`) but failed correctness at the active tolerance with `max_abs_error=7.18e-02`, so it was dropped.
- `block_fusion_902` stage-level swap: replacing the accepted stage3 block-only wrapper with a full-stage wrapper passed correctness but regressed the recognizer to `1.73 ms` on a `10/100` screen, so it was reverted.
- `block_fusion_912`: compile-only `net.backbone.stages.3.blocks.0` matched the current `1.40 ms` accepted latency on repeated `10/100` screens, so it was rejected as neutral even though the in-process reference measurement inflated during that swap test.
- `block_fusion_913`: compile-only `net.head.ctc_encoder.encoder` regressed the recognizer to `1.90 ms` on repeated `10/100` screens, so it was reverted.
- `graph_capture_914`: compiling the whole model before the outer CUDA graph replay recovered from the initial recursion bug but still regressed the accepted stack to `1.80 ms` on the `10/100` screen, so it was reverted.
- `block_fusion_915`: the stage2 preallocated-concat rewrite stayed bitwise exact but regressed the accepted stack to `1.50 ms` on the `10/100` screen, so it was reverted.
- `block_fusion_916`: compile-opaque `net.backbone` only recovered to `1.90 ms` on a reduced `layout + graph_capture + encoder` screen, so it was not promoted to the accepted-stack confirmation path.
- `block_fusion_917`: compile-opaque `net.backbone.stages.2` cleanly replaced the accepted stage2 stack after the dynamic kernel-module loader fix, but it still landed at `1.90 ms -> 1.90 ms` on the `10/100` screen, so it was reverted as neutral.
- `block_fusion_918`: compile-opaque `net.head.ctc_encoder` was bitwise exact but stayed at `1.90 ms -> 1.90 ms` on the `10/100` screen, so it was reverted as neutral.
- `block_fusion_919`: stage2 block-by-block compiled wrapper improved the `10/100` screen to `1.50 ms`, but the `20/400` confirmation was `1.41 ms -> 1.41 ms` (`0.996x`), so it was reverted.
- `block_fusion_920`: compile-original `net.head` improved the `10/100` screen to `1.40 ms`, but the `20/400` confirmation regressed slightly to `1.40 ms -> 1.41 ms` (`0.993x`), so it was reverted.
- `block_fusion_923`: compile-original `net` was only a `1.80 ms -> 1.79 ms` screen win (`1.006x`), so it was not promoted to `20/400` confirmation.
- `block_fusion_924`: the stage2 concat-free squeeze-accumulate rewrite preserved correctness but regressed the accepted stack to `1.80 ms -> 1.96 ms` (`0.918x`) on the `10/100` screen, so it was reverted immediately.
- `block_fusion_925`: compile-original `net.head.ctc_encoder` preserved the accepted inner encoder specialization and stayed exact, but the `20/400` confirmation only moved `1.33 ms -> 1.32 ms` (`1.002x`), so it was rejected as non-defensible.
- `block_fusion_926`: the SDPA-backed SVTR encoder rewrite fixed the earlier attention-path correctness issue and stayed within tolerance, but it was flat at `1.80 ms -> 1.80 ms` on the `10/100` screen, so it was rejected before `20/400` confirmation.
- `block_fusion_927`: a stage0 blocks-only wrapper that kept the downsample exact looked strong on the `10/100` screen, but the `20/400` confirmation flattened to `1.33 ms -> 1.32 ms` (`1.008x`), so it was rejected as non-defensible.
- `block_fusion_928`: inserting a direct stage0 block-level fold ahead of the accepted stage0 and backbone outer compiles also collapsed to `1.33 ms -> 1.32 ms` (`1.006x`) on `20/400`, so it was rejected.
- `graph_capture_929`: replaying into a reusable copied output buffer stayed bitwise exact and briefly looked better on the `10/100` screen, but it flattened to the same `1.33 ms -> 1.33 ms` bin on `20/400`, so it was rejected.
- `graph_capture_930`: adopting the caller's static input buffer and skipping the replay-time input copy also stayed bitwise exact and looked better on the short screen, but it flattened to the same `1.33 ms -> 1.33 ms` bin on `20/400`, so it was rejected.
- `block_fusion_931`: replacing the accepted `net.backbone` outer compile with a dedicated stage1-3 tail compile kept exact outputs, but it stayed flat at `1.8 ms -> 1.8 ms` on the `10/100` screen, so it was rejected as neutral.
- `block_fusion_932`: compiling only the SVTR token-mixer region inside `net.head.ctc_encoder.encoder` stayed within tolerance, but it landed at `1.8 ms -> 1.8 ms` (`0.99x`) on the `10/100` screen, so it was reverted.
- `block_fusion_933`: compiling each SVTR block privately inside the accepted encoder wrapper also stayed within tolerance, but it regressed slightly to `1.8 ms -> 1.8 ms` (`0.98x`) on the `10/100` screen, so it was reverted.
- `block_fusion_934`: a `max-autotune-no-cudagraphs` compile-original swap on `net.backbone` aligned with the refreshed live replay profile, but the autotuned schedules were much slower and regressed the screen from `1.4 ms` to `2.9 ms` (`0.48x`), so it was reverted immediately.
- `graph_capture_900` no-clone replay: reusing the static CUDA-graph output buffer without cloning passed correctness but regressed the recognizer to `2.27 ms` on a `10/100` screen, so it was reverted.

## Notes

- `block_fusion_908` supersedes `block_fusion_901` on `net.backbone.stages.2`; the verifier now resolves shared-target block-fusion conflicts by keeping the higher-priority candidate instead of stacking wrappers.
- `block_fusion_909` supersedes `block_fusion_906` on `net.head.ctc_encoder.encoder` through the same shared-target precedence path.
- `block_fusion_910` supersedes `block_fusion_905` on `net.backbone.stem` through the same shared-target precedence path.
- `block_fusion_911` supersedes the earlier stage0 folding attempt by switching that target to a compile-only wrapper, which preserved correctness while still giving Inductor a larger subgraph under the outer CUDA graph replay.
- `block_fusion_921` extends the accepted stage3 boundary from a single optimized block to the full stage while keeping the previously accepted `block_fusion_902` block0 specialization active inside the compiled wrapper.
- `block_fusion_922` shows that broader outer wrappers can still help when they enclose already-accepted inner specializations; this compile-original backbone wrapper outperformed the earlier reduced-stack compile-opaque backbone attempt.
- `block_fusion_908` and `block_fusion_902` structurally subsume the earlier rank-23 and rank-20 conv wrappers on their covered modules, so future verifier runs may report those conv candidates as skipped while still preserving the faster accepted stack.
- `load_kernel_module()` now registers dynamic kernel modules in `sys.modules`, which fixes TorchDynamo import failures when a compiled graph traces through already-loaded optimized kernels.
- `discover_optimized_kernels()` now ignores `reverted` entries from orchestration state even when their stale recorded speedup is still greater than `1.0`, which keeps rejected graph candidates from leaking back into `--workspace workspace/graph_capture_eval` acceptance runs.
- The accepted stack has moved past simple ConvBN folding. The latest `923/924` screens also suggest plain boundary compiles are close to exhausted; remaining work should favor deeper stage2 or encoder rewrites rather than more descriptor-level conv reinsertion.
- The `927/928` stage0 follow-ups show the same pattern as the head-side `925/926` work: once checked on `20/400`, the apparent screen wins collapse into a single `1.33 ms -> 1.32 ms` bin, so stage0 is effectively in the same exhausted bucket for now.
- `graph_capture_929` suggests the outer replay-time output clone is not the dominant remaining cost on the accepted stack. Reusing a copied output buffer changed the replacement set slightly, but it did not move the defended `20/400` latency bin.
- `graph_capture_930` closes the other obvious replay-time lever too: adopting the caller-owned static input buffer and skipping the replay-time input copy still did not move the defended `20/400` latency bin.
- The `931/932/933` follow-ups close the other obvious structural rewrites on the accepted stack. A dedicated backbone-tail compile and two finer-grained SVTR compile splits were all exact, but none moved the `10/100` recognizer bin.
- `block_fusion_934` shows the remaining replay-path cost is not fixable by simply asking Inductor to search harder on the current outer backbone boundary. The live stack is already dominated by inner conv and GEMM kernels, and the max-autotune schedule search made that boundary materially worse.

## Refreshed Profile

Section microprofile on the accepted structural stack without CUDA graph replay, benchmarking each live module on the real tensors it receives during a model forward. After `block_fusion_922`, the useful ranking boundary is the whole backbone plus the head path:

- `backbone`: `5.37 ms`
- `head`: `1.35 ms`
- `head.ctc_encoder`: `1.25 ms`
- `head.ctc_encoder.encoder`: `1.22 ms`
- `head.ctc_head`: `0.10 ms`

Deeper encoder microprofile on the accepted structural stack after the `925/926` refresh, still without outer CUDA graph replay:

- `head.ctc_encoder.encoder._orig_mod.svtr_block.0`: `0.95 ms`
- `head.ctc_encoder.encoder._orig_mod.svtr_block.0.mixer`: `0.45 ms`
- `head.ctc_encoder.encoder._orig_mod.svtr_block.1`: `0.89 ms`
- `head.ctc_encoder.encoder._orig_mod.svtr_block.1.mixer`: `0.40 ms`
- `head.ctc_encoder.encoder._orig_mod.conv1/2/3/4/1x1`: each `0.15-0.17 ms`

Latest structural microprofile after the `931-934` loop, still on the accepted stack without outer CUDA graph replay and using the real tensors captured at each module boundary:

- `net.backbone`: `5.63 ms`
- `net.backbone._orig_mod.stages.2`: `2.94 ms`
- `net.head`: `1.38 ms`
- `net.head.ctc_encoder`: `1.42 ms`
- `net.head.ctc_encoder.encoder`: `1.25 ms`
- `net.backbone._orig_mod.stages.3`: `1.16 ms`
- `net.backbone._orig_mod.stages.0`: `0.78 ms`
- `net.backbone._orig_mod.stages.1`: `0.76 ms`
- `net.head.ctc_encoder.encoder._orig_mod.svtr_block.0`: `0.70 ms`
- `net.head.ctc_encoder.encoder._orig_mod.svtr_block.1`: `0.70 ms`

Live replay-path profile on the accepted graph-captured stack itself, using 10 profiled replays and sorting CUDA self time totals:

- `sm75_xmma_fprop_implicit_gemm...64x64x64...`: `4.81 ms` total across 10 replays
- `conv2d_c1_k1_nhwc`: `2.43 ms`
- grouped direct cuDNN conv kernel: `2.05 ms`
- `sm75_xmma_fprop_implicit_gemm...64x32x64...`: `1.54 ms`
- `sm75_xmma_fprop_implicit_gemm...64x64x64...`: `1.32 ms`
- indexed implicit GEMM cuDNN conv: `0.81 ms`
- best remaining attention-side softmax kernel: `0.12 ms`

## Fresh Plan

- The accepted stack still points to the same coarse structural hotspots without the outer graph, but the live replay profile now shows the defended `1.33 ms` bin is dominated by inner conv and GEMM kernels rather than extra Python or graph-launch overhead.
- The `931/932/933` experiments exhausted the next obvious structure-only boundary changes on top of `block_fusion_922`: dedicated backbone-tail compile, token-mixer-only compile, and per-SVTR-block compile all stayed exact but flat.
- The `934` experiment closed the compile-mode branch too. Asking Inductor to autotune the accepted backbone boundary searched many conv and GEMM schedules, but it regressed badly instead of uncovering a new structural win.
- With the current constraint against spending time on standalone conv or batchnorm kernels, graph-level work is at a hard blocker on this stack. The remaining defended replay-path headroom appears to be inside per-kernel conv/GEMM implementations, not in another profitable block or CUDA-graph boundary.
- A future restart should either relax the “no standalone conv/GEMM work” constraint for the replay-path hotspots identified above, or bring in a genuinely different graph-level mechanism than compile-boundary reshaping on the existing accepted stack.
