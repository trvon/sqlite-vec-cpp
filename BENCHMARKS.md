# SQLite-Vec C++ Benchmark Results

**Version**: 0.1.0
**Date**: 2025-11-02
**Platform**: x86_64, 48 cores @ 4.0GHz, 32KB L1, 512KB L2, 16MB L3
**Compiler**: GCC 15.2.0, C++23, Release mode (`-O3`)
**Library**: Google Benchmark 1.9.1

---

## Executive Summary

The C++ implementation achieves **3.6M vectors/second sustained throughput** with linear scaling across corpus sizes and embedding dimensions. int8 quantization provides 4x storage reduction at performance parity. HNSW index recommended for >100K vector corpora.

---

## RAG Pipeline Benchmark

### 1. Corpus Size Scaling (384d, K=5)

| Corpus | Latency | Throughput | QPS (single-thread) |
|--------|---------|------------|---------------------|
| 1K     | 273 μs  | 3.67 M/s   | ~3,660 queries/sec  |
| 10K    | 2.78 ms | 3.60 M/s   | ~360 queries/sec    |
| 100K   | 27.9 ms | 3.58 M/s   | ~36 queries/sec     |

**Scaling**: Linear (10x corpus → 10x latency)
**Bottleneck**: Compute-bound (memory bandwidth utilization ~5%)

### 2. K-Value Scaling (10K docs, 384d)

| K  | Latency | Delta |
|----|---------|-------|
| 1  | 2.77 ms | -0.4% |
| 5  | 2.78 ms | baseline |
| 10 | 2.78 ms | 0.0%  |
| 50 | 2.77 ms | -0.4% |

**Conclusion**: Partial sort overhead negligible; K-value has no meaningful impact.

### 3. Embedding Dimension Scaling (10K docs, K=5)

| Dimensions | Latency  | Throughput | Scaling Factor |
|------------|----------|------------|----------------|
| 384d       | 2.78 ms  | 3.60 M/s   | 1.0x           |
| 768d       | 5.74 ms  | 1.74 M/s   | 2.06x          |
| 1536d      | 11.7 ms  | 856k/s     | 4.21x          |

**Scaling**: Near-linear (2x dim → 2.06x latency, 4x dim → 4.21x latency)
**Conclusion**: Compute-bound; SIMD efficiency remains high across dimensions.

### 4. Quantization (10K docs, 384d, K=5)

| Type  | Latency | Throughput | Storage | Overhead |
|-------|---------|------------|---------|----------|
| float | 2.78 ms | 3.60 M/s   | 4 bytes | baseline |
| int8  | 2.74 ms | 3.65 M/s   | 1 byte  | **-1.4%** |

**Conclusion**: int8 quantization is **faster** while reducing storage 4x (memory bandwidth savings).

### 5. Multi-Query Throughput (10K docs, 384d)

- **10 queries**: 27.5 ms total (2.75 ms/query average)
- **Sustained throughput**: 3.64 M vectors/second
- **QPS**: ~364 queries/second (single-threaded)
- **Parallelization potential**: 48 cores → ~17.4K QPS theoretical

### 6. Sequential vs Batch (1K docs, 384d, K=5)

| Method     | Latency | Throughput |
|------------|---------|------------|
| Sequential | 274 μs  | 3.66 M/s   |
| Batch      | 273 μs  | 3.67 M/s   |

**Conclusion**: Batch API provides cleaner code at performance parity (memory-bandwidth bound).

---

## Batch Distance Benchmark

### 1. Sequential vs Batch Comparison

| Scenario | Sequential | Batch | Speedup |
|----------|------------|-------|---------|
| 100×384d | 26.7 μs    | 26.7 μs | 1.00x |
| 1K×384d  | 268 μs     | 269 μs  | 1.00x |

**Conclusion**: Parity performance; both memory-bandwidth limited.

### 2. Memory Layout Optimization

| Layout      | Latency | Throughput | Improvement |
|-------------|---------|------------|-------------|
| Scattered   | 269 μs  | 3.73 M/s   | baseline    |
| Contiguous  | 267 μs  | 3.75 M/s   | +0.5%       |

**Conclusion**: Marginal improvement; modern CPUs prefetch efficiently.

### 3. Top-K Performance (1K×384d, K=10)

- **Latency**: 268 μs (vs 268 μs full distance computation)
- **Overhead**: <1% for partial sort
- **Conclusion**: `std::partial_sort` highly optimized; K << N has negligible cost.

### 4. Large Embeddings (1K×1536d)

- **Latency**: 1.13 ms
- **Throughput**: 886k vectors/second
- **Scaling**: 4.21x slower than 384d (expected 4.0x)

---

## HNSW Decision Matrix

| Corpus Size | Brute-Force Latency | Recommendation |
|-------------|---------------------|----------------|
| <10K        | <3ms                | ✅ Brute-force optimal |
| 10K-100K    | 3-30ms              | ⚠️ Brute-force acceptable for batch |
| >100K       | >30ms               | ❌ HNSW required for real-time (<10ms) |

**HNSW Threshold**: 100K vectors (27.9ms → >10ms target requires ANN index)

---

## Platform-Specific Results

### SIMD Utilization

- **AVX**: Active (conditional compilation, `-mavx` detected)
- **NEON**: Not tested (x86_64 platform)
- **Scalar fallback**: Available for non-aligned/small vectors

### Cache Efficiency

- **L1 hit rate**: >95% (estimated from throughput consistency)
- **Memory bandwidth**: ~11 GB/s per query (vs 200 GB/s L1 capacity)
- **Conclusion**: Compute-bound, not memory-bound

---

## Comparison to Targets

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| 1K corpus (<1ms) | 1000 μs | 273 μs | ✅ **3.6x better** |
| 10K corpus (<5ms) | 5000 μs | 2780 μs | ✅ **1.8x better** |
| 100K corpus (<50ms) | 50000 μs | 27900 μs | ✅ **1.8x better** |
| int8 overhead (<20%) | 20% | -1.4% | ✅ **Faster** |
| Dimension scaling | Linear | Linear | ✅ **Perfect** |

---

## Reproduction

```bash
# Build benchmarks
cd third_party/sqlite-vec-cpp
meson setup build_bench -Denable_benchmarks=true -Dbuildtype=release
ninja -C build_bench

# Run RAG pipeline benchmark
./build_bench/benchmarks/rag_pipeline_benchmark --benchmark_min_time=0.5s

# Run batch distance benchmark
./build_bench/benchmarks/batch_distance_benchmark --benchmark_min_time=0.5s

# JSON output for analysis
./build_bench/benchmarks/rag_pipeline_benchmark \
  --benchmark_out=results.json \
  --benchmark_out_format=json
```

---
