# SQLite-Vec C++ Benchmark Results

**Version**: 0.1.0
**Date**: 2026-01-19
**Platform**: Apple M3 Max, 16 cores, 48 GB RAM (macOS 26.2)
**Compiler**: Apple clang 17.0.0, C++23, Release mode (`-O3` via Meson `buildtype=release`)
**SIMD**: NEON enabled, ARM DotProd enabled
**Library**: Google Benchmark 1.8.3

> Note: Google Benchmark reports “Library was built as DEBUG” even in this Release build; the Meson
> configuration is `buildtype=release` with NEON/DotProd enabled.

---

## Archive

Previous benchmark runs are archived in `benchmarks/archive/`.

---

## Batch Distance Benchmark

### 1. Sequential vs Batch Comparison

| Scenario | Time | Throughput |
|----------|------|------------|
| 100×384d (Sequential) | 2.383 µs | 41.96 M/s |
| 100×384d (Batch)      | 2.523 µs | 39.64 M/s |
| 1K×384d (Sequential)  | 25.50 µs | 39.21 M/s |
| 1K×384d (Batch)       | 26.01 µs | 38.45 M/s |

### 2. Memory Layout Optimization

| Layout | Time | Throughput |
|--------|------|------------|
| Contiguous (1K×384d) | 22.75 µs | 43.96 M/s |

### 3. Top‑K Performance (1K×384d, K=10)

- **Latency**: 26.85 µs
- **Throughput**: 37.25 M/s

### 4. Quantization (1K×384d)

| Type | Time | Throughput |
|------|------|------------|
| int8 | 38.22 µs | 26.16 M/s |

### 5. Large Embeddings (1K×1536d)

- **Latency**: 110.9 µs
- **Throughput**: 9.02 M/s

---

## RAG Pipeline Benchmark

### 1. Corpus Size Scaling (384d, K=5)

| Corpus | Latency | Throughput |
|--------|---------|------------|
| 1K     | 28.3 µs | 35.35 M/s  |
| 10K    | 253 µs  | 39.51 M/s  |
| 100K   | 5.67 ms | 17.64 M/s  |

### 2. K‑Value Scaling (10K docs, 384d)

| K  | Latency | Throughput |
|----|---------|------------|
| 1  | 305 µs  | 32.82 M/s  |
| 5  | 253 µs  | 39.51 M/s  |
| 10 | 254 µs  | 39.43 M/s  |
| 50 | 342 µs  | 29.20 M/s  |

### 3. Embedding Dimension Scaling (10K docs, K=5)

| Dimensions | Latency | Throughput | Scaling Factor |
|------------|---------|------------|----------------|
| 384d       | 253 µs  | 39.51 M/s  | 1.00x |
| 768d       | 740 µs  | 13.51 M/s  | 2.92x |
| 1536d      | 1122 µs | 8.91 M/s   | 4.43x |

### 4. Quantization (10K docs, 384d, K=5)

| Type  | Latency | Throughput |
|-------|---------|------------|
| float | 253 µs  | 39.51 M/s  |
| int8  | 414 µs  | 24.13 M/s  |

### 5. Multi‑Query Throughput (10K docs, 384d, 10 queries)

- **Total time**: 3.01 ms
- **Throughput**: 33.23 M/s

---

## Filtered Search Benchmark (HNSW, 10K corpus)

| Scenario | Time | Throughput |
|----------|------|------------|
| No filter | 9.83 ms | 10.17 k/s |
| Bitset filter 10% | 50.93 ms | 1.96 k/s |
| Bitset filter 50% | 19.42 ms | 5.15 k/s |
| Bitset filter 90% | 11.07 ms | 9.03 k/s |
| Set filter 10% | 65.46 ms | 1.53 k/s |
| Set filter 50% | 23.39 ms | 4.28 k/s |
| Set filter 90% | 11.78 ms | 8.49 k/s |

---

## HNSW Index Performance

Full HNSW benchmark run was stopped due to long runtime. Partial results are logged in
`benchmarks/logs/2026-01-19_release_neon/hnsw_benchmark.log`. We will update this section after
optimizing the long‑running benchmark and re‑running.

---

## Reproducibility

Release build with NEON and DotProd:

```
meson setup builddir-release -Dbuildtype=release -Denable_benchmarks=true -Denable_simd_neon=true
meson compile -C builddir-release
./builddir-release/benchmarks/batch_distance_benchmark
./builddir-release/benchmarks/rag_pipeline_benchmark
./builddir-release/benchmarks/filtered_search_benchmark
```

Logs are stored under `benchmarks/logs/2026-01-19_release_neon/`.
