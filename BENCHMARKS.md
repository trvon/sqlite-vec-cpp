# Vector Engine Benchmarks

Apple M4 Max, 16-core, macOS 26.5. YAMS debug build with asan+coverage.

## Search engine comparison (vector_backend_engine_compare)

2000 normalized random vectors, 50 queries, 128 dimensions, k=10.

| Engine | Build (ms) | Query mean (us) | Query p95 (us) | recall@10 |
|--------|-----------|----------------|---------------|-----------|
| simeon-pq | 3,574 | 1,501 | 1,553 | 73.6% |
| vec0-l2 | 24 | 2,558 | 2,731 | 99.8% |

Note: vec0-l2 mean is inflated by first-query HNSW build; evaluate p95 for steady-state.

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
