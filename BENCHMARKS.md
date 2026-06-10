# Vector Engine Benchmarks

Apple M4 Max, 16-core, macOS 26.5. YAMS debug build with asan+coverage.

## Quantized two-stage HNSW search (quantized_search_benchmark)

10,000 Gaussian vectors, 100 queries, k=10, debugoptimized build. True RaBitQ
(arXiv:2405.12497: FWHT rotation + unbiased estimator + 4-bit query planes)
replaced the previous sign-quantization heuristic in 2026-06.

### 128d, ef_search=100

| Method | Latency (us) | Recall@10 | Quant Memory |
|--------|--------------|-----------|--------------|
| FP32 baseline | 214 | 98.1% | 0 B |
| LVQ-8 (3x rerank) | 430 | 99.8% | 1,360,000 B |
| LVQ-4 (3x rerank) | 435 | 99.9% | 720,000 B |
| RaBitQ (3x rerank) | 437 | 94.9% | 240,544 B |
| RaBitQ (5x rerank) | 609 | 98.0% | 240,544 B |

Previous sign-quantization RaBitQ at this cell: 76.6% (3x) / 85.5% (5x).

### 768d, ef_search=100/200

| Method | Latency (us) | Recall@10 | Quant Memory |
|--------|--------------|-----------|--------------|
| FP32 baseline (ef=100) | 921 | 90.2% | 0 B |
| LVQ-8 (3x, ef=100) | 1,430 | 99.3% | 7,760,000 B |
| LVQ-4 (3x, ef=100) | 1,470 | 99.4% | 3,920,000 B |
| RaBitQ (3x, ef=100) | 733 | 96.7% | 1,363,328 B |
| RaBitQ (5x, ef=100) | 1,233 | 99.3% | 1,363,328 B |
| RaBitQ (3x, ef=200) | 1,152 | 99.6% | 1,363,328 B |
| RaBitQ (5x, ef=200) | 1,586 | 100.0% | 1,363,328 B |

At 768d RaBitQ dominates the latency/recall/memory frontier: popcount cost
scales with D/8 while LVQ scales with D. Extended (multi-bit) RaBitQ was
evaluated as a follow-up and deferred: 1-bit codes + FP32 rerank already match
LVQ-8 recall at 5.7x less memory, so extra bits buy nothing in the two-stage
configuration.

## Search engine comparison (vector_backend_engine_compare)

2000 normalized random vectors, 50 queries, 128 dimensions, k=10.

| Engine | Build (ms) | Query mean (us) | Query p95 (us) | recall@10 |
|--------|-----------|----------------|---------------|-----------|
| simeon-pq | 3,574 | 1,501 | 1,553 | 73.6% |
| vec0-l2 | 24 | 2,558 | 2,731 | 99.8% |

Note: vec0-l2 mean is inflated by first-query HNSW build; evaluate p95 for steady-state.

### simeon-pq rerank fix (2026-06-09)

The historical 73.6% recall was caused by a rerank bug in
`simeonPqSearchUnlocked`: candidates were selected in approximate-ADC order
and the exact cosine was computed for display only, so `rerank_factor` had no
effect on selection. After the fix (exact-score all `k*rerank` candidates,
sort by exact similarity, take top-k), the `--spq-sweep` cell at the same
2K/128d configuration measures:

| m | rerank | recall@10 | mean (us) |
|---|--------|-----------|-----------|
| 32 | 2 (default) | 94.0% | 14,535 |
| 32 | 4 | 99.4% | 18,693 |
| 32 | 8 | 100.0% | 23,668 |
| 16 | 8 | 96.0% | 20,161 |
| 64 | 2 | 100.0% | 21,314 |

Pre-fix, every rerank row measured identically (73.6% at m=32). Remaining
latency is the brute-force ADC scan plus per-candidate record fetch, not the
rerank stage.

At 768d on random Gaussian vectors (PQ worst case, no structure), the fix
makes rerank effective (m=8: 16.0% → 37.2% across rerank 2→8) but absolute
recall stays low (m=32/rerank=2: 34.4%; m=64/rerank=4: 67.8%) — consistent
with the existing PQ-recall-vs-dimension section. For ≥768d corpora prefer
the vec0-l2 engine or m≥64 with rerank≥4; judge real (non-Gaussian) corpora
separately before changing defaults.

## HNSW delete-policy churn (hnsw_churn_benchmark)

10K Gaussian vectors, 128d, 10 cycles of (delete 500 + insert 500), k=10,
ef_search=100. Recall measured against brute force over the active set;
`fresh` is a from-scratch build of the final active set.

| Policy | Recall@10 range (cycles 1-10) | Latency growth | Churn cost/cycle |
|--------|------------------------------|----------------|------------------|
| soft (default) | 98.4% – 99.2% | 230 → 492 us | ~1.4 s |
| isolate_deleted | 93.0% – 97.4% (degrading) | 260 → 324 us | ~1.0 s |
| compact each cycle | 96.0% – 97.8% | stable | 19 – 31 s |
| fresh reference | 97.2% | 222 us | — |

Verdict: soft deletion (retaining deleted nodes as traversal waypoints)
*exceeds* fresh-built recall through 50% corpus turnover, so MN-RU-style
repair-on-delete (arXiv:2407.07871) is unnecessary for the default policy —
it only addresses the degradation that `isolate_deleted()` introduces.
The costs of soft deletion are graph growth (memory, ~2x latency at 50%
turnover), which periodic `compact()` at the `needs_compaction()` threshold
already bounds.

## Build-time scaling by dimension (2000 vectors)

| Engine | 128d (ms) | 384d (ms) | 1024d (ms) |
|--------|----------|----------|-----------|
| simeon-pq | 3,592 | 8,327 | 19,032 |
| vec0-l2 | 42 | 62 | 75 |

## Query latency by dimension (p95, 2000 vectors)

| Engine | 128d (us) | 384d (us) | 1024d (us) |
|--------|----------|----------|-----------|
| simeon-pq | 1,553 | — | — |
| vec0-l2 | 2,731 | — | — |

## Simeon encoder throughput (4096→384 projection, asan build)

| Projection | us/doc | docs/sec | vs Achlioptas |
|-----------|--------|---------|--------------|
| FWHT | 473 | 2,113 | 13.2× |
| VerySparse | 360 | 2,776 | 17.3× |
| DenseGaussian | 5,100 | 196 | 1.2× |
| AchlioptasSparse | 6,237 | 160 | 1.0× |

FWHT cost is nearly constant across output dimensions (456–473 us/doc for 256–768d output), consistent with the O(n log n + k) FJLT complexity.

## vec0 ANN standalone benchmark

2000 vectors, 128 dimensions, 50 queries, k=10. No index prebuild.

Warm ANN first-query build: 985 ms. Warm search: 62 us/query, 16,042 QPS, 100% recall@10.

## PQ recall vs dimension (m=32, 2000 vectors)

| Dimension | recall@10 |
|-----------|----------|
| 128 | 73.6% |
| 256 | 49.8% |
| 1024 | 51.0% |

Subvector dimension at m=32 is dim/32. Literature suggests subdim 8–16 for recall-sensitive use; m=32 gives subdim=32 at 1024d, undersized for production recall targets.
