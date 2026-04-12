# SQLite-Vec C++ Benchmark Results

**Version**: 0.1.0
**Date**: 2026-04-12
**Platform**: Apple M4 Max, 16 cores (macOS 26.4)
**Compiler**: Apple clang 17.0.0, C++20, Release mode (`-O3` via Meson `buildtype=release`)
**SIMD**: NEON enabled, ARM DotProd disabled
**Library**: Google Benchmark 1.9.5

> Note: These benchmark settings are intentionally tuned for local performance measurement.
> They are not the recommended defaults for packaging or portable distribution builds.
>
> The Jan 2026 baseline ran on Apple M3 Max with DotProd enabled. This run uses Apple M4 Max
> without DotProd. Latency comparisons reflect both hardware and code changes; recall
> comparisons are hardware-independent.

---

## Archive

Previous benchmark runs are archived in `benchmarks/archive/`.

---

## HNSW Engine Comparison Benchmark (YAMS baseline + optional zvec)

This benchmark is implemented by `benchmarks/hnsw_engine_comparison_benchmark.cpp` and built as
`hnsw_engine_comparison_benchmark`.

Current status for this report: **YAMS-only run** (zvec was not linked in this run).

### What it measures

- Index build time
- Search latency and QPS at `ef_search` values 50, 100, 200
- Recall@K against brute-force ground truth

### Build and run

From `third_party/sqlite-vec-cpp/`:

```bash
meson setup builddir
meson compile -C builddir

# YAMS baseline (default)
./builddir/benchmarks/hnsw_engine_comparison_benchmark --corpus=10000 --dim=768
```

Optional zvec-enabled comparison:

```bash
# build zvec separately
git clone https://github.com/alibaba/zvec.git /opt/zvec
cd /opt/zvec && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# reconfigure sqlite-vec-cpp with zvec headers
cd /path/to/yams/third_party/sqlite-vec-cpp
meson setup builddir -Dzvec-root=/opt/zvec --reconfigure
meson compile -C builddir
./builddir/benchmarks/hnsw_engine_comparison_benchmark --corpus=10000 --dim=768
```

### Current run results (2026-04-12, Apple M4 Max)

Run: `./hnsw_engine_comparison_benchmark --corpus=10000 --dim=768`

| Engine | M | ef_search | Build (ms) | Latency (us) | QPS | Recall@10 |
|--------|---|-----------|------------|--------------|-----|-----------|
| yams-hnsw | 16 | 50 | 38,863 | 502 | 1,991 | 54.1% |
| yams-hnsw | 16 | 100 | 38,863 | 843 | 1,186 | 74.7% |
| yams-hnsw | 16 | 200 | 38,863 | 1,308 | 765 | 92.9% |
| yams-hnsw | 24 | 50 | 83,918 | 721 | 1,386 | 68.7% |
| yams-hnsw | 24 | 100 | 83,918 | 1,116 | 896 | 86.9% |
| yams-hnsw | 24 | 200 | 83,918 | 1,643 | 609 | 98.4% |
| yams-hnsw | 32 | 50 | 144,767 | 949 | 1,054 | 77.4% |
| yams-hnsw | 32 | 100 | 144,767 | 1,347 | 743 | 93.6% |
| yams-hnsw | 32 | 200 | 144,767 | 1,873 | 534 | 99.5% |

Read-only mode (`yams-hnsw-ro`) produces equivalent latency and recall.

Parallel build (M=24, ef_c=200): **6,785 ms** (11.2x speedup over sequential, 16 threads).

Hardware note: Apple M4 Max, NEON (no DotProd), FP32, single-threaded query loop.

### Comparison vs Jan 2026 baseline (M3 Max)

> The Jan baseline ran on Apple M3 Max with DotProd enabled. This run uses Apple M4 Max
> without DotProd. Latency deltas reflect both hardware differences and code optimizations
> (prefetch hints, zero-copy neighbor traversal, flat dense_id lookup, visited pool bitmap).
> Recall is deterministic and hardware-independent — identical recall confirms no algorithmic
> regressions.

| M | ef_search | Jan Latency (M3) | Apr Latency (M4) | Delta | Recall |
|---|-----------|-------------------|-------------------|-------|--------|
| 16 | 50 | 693 us | 502 us | **-27.6%** | 54.1% (unchanged) |
| 16 | 100 | 1,129 us | 843 us | **-25.3%** | 74.7% (unchanged) |
| 16 | 200 | 1,819 us | 1,308 us | **-28.1%** | 92.9% (unchanged) |
| 24 | 50 | 1,028 us | 721 us | **-29.9%** | 68.2% -> 68.7% |
| 24 | 100 | 1,529 us | 1,116 us | **-27.0%** | 86.9% (unchanged) |
| 24 | 200 | 2,219 us | 1,643 us | **-26.0%** | 98.5% -> 98.4% |
| 32 | 100 | 1,811 us | 1,347 us | **-25.6%** | 93.1% -> 93.6% |
| 32 | 200 | 2,405 us | 1,873 us | **-22.1%** | 99.5% (unchanged) |

### External zvec reference numbers (not measured in this run)

From zvec published benchmarks (<https://zvec.org/en/docs/benchmarks/>):

| Dataset | Config | QPS | Recall |
|---------|--------|-----|--------|
| Cohere 1M (768d) | INT8, M=15, ef_search=180 | ~16,000 | 95%+ |
| Cohere 10M (768d) | INT8, M=50, ef_search=118, refiner | ~8,000 | 95%+ |

Key differences vs this run:

- zvec numbers use INT8 + refiner and multithreaded query load
- this run uses FP32 and single-threaded query loop
- hardware differs (cloud ECS vs local Apple Silicon)

---

## Quantized HNSW Search Benchmark (NEW)

This benchmark is implemented by `benchmarks/quantized_search_benchmark.cpp` and built as
`quantized_search_benchmark`. It measures two-stage quantized search: approximate distance
computation using quantized codes for HNSW traversal, followed by exact FP32 reranking.

### What it measures

- Build time for quantization codes
- Search latency and QPS at various ef_search values
- Recall@K against brute-force ground truth
- Memory usage of quantized stores vs FP32 vectors

### Build and run

```bash
./builddir/benchmarks/quantized_search_benchmark --corpus 5000 --dim 384 --queries 100
```

### Current run results (2026-04-12, Apple M4 Max)

Corpus: 5000 vectors, 384d, 100 queries, k=10

#### ef_search = 50

| Method | Build (ms) | Latency (us) | QPS | Recall@10 | Quant Memory |
|--------|------------|--------------|-----|-----------|--------------|
| FP32 baseline | 0.8 | 139.0 | 7,192 | 91.0% | 0 B |
| LVQ-8 (2x rerank) | 1.9 | 246.8 | 4,052 | 98.1% | 1,960,000 B |
| LVQ-8 (3x rerank) | 1.9 | 314.1 | 3,184 | 99.5% | 1,960,000 B |
| LVQ-4 (3x rerank) | 3.1 | 305.2 | 3,276 | 99.4% | 1,000,000 B |
| RaBitQ (3x rerank) | 9.3 | 171.4 | 5,834 | 75.5% | 261,536 B |
| RaBitQ (5x rerank) | 9.4 | 239.9 | 4,169 | 86.3% | 261,536 B |

#### ef_search = 100

| Method | Build (ms) | Latency (us) | QPS | Recall@10 | Quant Memory |
|--------|------------|--------------|-----|-----------|--------------|
| FP32 baseline | 0.5 | 240.7 | 4,155 | 98.2% | 0 B |
| LVQ-8 (2x rerank) | 1.7 | 361.2 | 2,769 | 99.9% | 1,960,000 B |
| LVQ-8 (3x rerank) | 1.7 | 446.4 | 2,240 | 100.0% | 1,960,000 B |
| LVQ-4 (3x rerank) | 3.3 | 430.9 | 2,321 | 100.0% | 1,000,000 B |
| RaBitQ (3x rerank) | 9.4 | 249.6 | 4,006 | 88.5% | 261,536 B |
| RaBitQ (5x rerank) | 9.6 | 393.8 | 2,539 | 93.7% | 261,536 B |

#### ef_search = 200

| Method | Build (ms) | Latency (us) | QPS | Recall@10 | Quant Memory |
|--------|------------|--------------|-----|-----------|--------------|
| FP32 baseline | 0.5 | 321.4 | 3,111 | 99.9% | 0 B |
| LVQ-8 (2x rerank) | 1.8 | 530.6 | 1,885 | 100.0% | 1,960,000 B |
| LVQ-8 (3x rerank) | 1.8 | 712.3 | 1,404 | 100.0% | 1,960,000 B |
| LVQ-4 (3x rerank) | 3.1 | 749.5 | 1,334 | 100.0% | 1,000,000 B |
| RaBitQ (3x rerank) | 9.5 | 526.2 | 1,900 | 94.8% | 261,536 B |
| RaBitQ (5x rerank) | 9.4 | 758.7 | 1,318 | 98.4% | 261,536 B |

FP32 vector memory: 7,680,000 bytes (7.3 MB).

### Compression ratios

| Method | Memory | Compression vs FP32 |
|--------|--------|---------------------|
| LVQ-8 | 1.96 MB | 3.9x |
| LVQ-4 | 1.00 MB | 7.7x |
| RaBitQ | 0.26 MB | 29.4x |

### Key findings

- **LVQ-8 (2x rerank)** at ef_search=50: 98.1% recall at 247 us (7.1% more recall than FP32 baseline at 1.8x latency cost)
- **LVQ-4 (3x rerank)**: matches LVQ-8 latency (305 us) with 7.7x compression (NEON-optimized nibble unpacking)
- **RaBitQ**: lowest memory (29.4x compression) but lower recall; best for memory-constrained deployments
- At ef_search=100, both LVQ-8 and LVQ-4 achieve **100% recall@10**

---

## Batch Distance Benchmark

### 1. Sequential vs Batch Comparison

| Scenario | Time | Throughput |
|----------|------|------------|
| 100x384d (Sequential) | 2.222 us | 45.01 M/s |
| 100x384d (Batch)      | 2.201 us | 45.42 M/s |
| 1Kx384d (Sequential)  | 24.04 us | 41.60 M/s |
| 1Kx384d (Batch)       | 24.03 us | 41.61 M/s |

### 2. Memory Layout Optimization

| Layout | Time | Throughput |
|--------|------|------------|
| Contiguous (1Kx384d) | 21.32 us | 46.90 M/s |

### 3. Top-K Performance (1Kx384d, K=10)

- **Latency**: 25.37 us
- **Throughput**: 39.41 M/s

### 4. Quantization (1Kx384d)

| Type | Time | Throughput |
|------|------|------------|
| int8 | 16.24 us | 61.57 M/s |

### 5. Large Embeddings (1Kx1536d)

- **Latency**: 96.63 us
- **Throughput**: 10.35 M/s

---

## RAG Pipeline Benchmark

### 1. Corpus Size Scaling (384d, K=5)

| Corpus | Latency | Throughput |
|--------|---------|------------|
| 1K     | 26.6 us | 37.64 M/s  |
| 10K    | 238 us  | 41.95 M/s  |
| 100K   | 5.68 ms | 17.60 M/s  |

### 2. K-Value Scaling (10K docs, 384d)

| K  | Latency | Throughput |
|----|---------|------------|
| 1  | 376 us  | 26.61 M/s  |
| 5  | 238 us  | 41.95 M/s  |
| 10 | 313 us  | 31.91 M/s  |
| 50 | 240 us  | 41.61 M/s  |

### 3. Embedding Dimension Scaling (10K docs, K=5)

| Dimensions | Latency | Throughput | Scaling Factor |
|------------|---------|------------|----------------|
| 384d       | 238 us  | 41.95 M/s  | 1.00x |
| 768d       | 810 us  | 12.35 M/s  | 3.40x |
| 1536d      | 1088 us | 9.19 M/s   | 4.57x |

### 4. Quantization (10K docs, 384d, K=5)

| Type  | Latency | Throughput |
|-------|---------|------------|
| float | 238 us  | 41.95 M/s  |
| int8  | 159 us  | 62.81 M/s  |

### 5. Multi-Query Throughput (10K docs, 384d, 10 queries)

- **Total time**: 2.32 ms
- **Throughput**: 43.15 M/s

---

## Filtered Search Benchmark (HNSW, 10K corpus)

| Scenario | Time | Throughput |
|----------|------|------------|
| No filter | 10.75 ms | 9.30 k/s |
| Bitset filter 10% | 54.72 ms | 1.83 k/s |
| Bitset filter 50% | 19.96 ms | 5.01 k/s |
| Bitset filter 90% | 11.35 ms | 8.81 k/s |
| Set filter 10% | 66.86 ms | 1.50 k/s |
| Set filter 50% | 24.12 ms | 4.15 k/s |
| Set filter 90% | 11.29 ms | 8.86 k/s |

---

## HNSW Index Performance

Full HNSW benchmark run was stopped due to long runtime. Partial results are logged in
`benchmarks/logs/2026-04-12_post-quantization/hnsw_benchmark.log`. We will update this section after
optimizing the long-running benchmark and re-running.

---

## Reproducibility

Release build with NEON:

```
meson setup builddir-release -Dbuildtype=release -Denable_benchmarks=true -Denable_simd_neon=true
meson compile -C builddir-release
./builddir-release/benchmarks/batch_distance_benchmark
./builddir-release/benchmarks/rag_pipeline_benchmark
./builddir-release/benchmarks/filtered_search_benchmark
./builddir-release/benchmarks/quantized_search_benchmark --corpus 5000 --dim 384 --queries 100
```

Logs are stored under `benchmarks/logs/2026-04-12_post-quantization/`.

For the HNSW engine comparison benchmark:

```bash
./builddir-release/benchmarks/hnsw_engine_comparison_benchmark --corpus=10000 --dim=768
```
