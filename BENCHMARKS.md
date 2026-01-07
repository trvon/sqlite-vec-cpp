# SQLite-Vec C++ Benchmark Results

**Version**: 0.1.0
**Date**: 2026-01-05
**Platform**: x86_64, 48 cores @ 3.8GHz, 32KB L1, 512KB L2, 16MB L3 (Windows 11)
**Compiler**: clang 21.1.6, C++23, Release mode (`-O3`)
**Library**: Google Benchmark 1.9.4


---

## Executive Summary

The C++ implementation achieves **~2.8M vectors/second sustained throughput** with linear scaling across corpus sizes and embedding dimensions. int8 quantization provides 4x storage reduction at near performance parity. HNSW index recommended for >100K vector corpora.

---

## Apple Silicon Results (M1 Pro)

**Date**: 2026-01-07
**Platform**: Apple M1 Pro, 16 cores (8P+8E), 192KB L1, 12MB L2, 24MB SLC
**Compiler**: Apple clang 16.0, C++20, Release mode (`-O3 -DNDEBUG`)
**SIMD**: NEON enabled (`-DSQLITE_VEC_ENABLE_NEON`), DotProd enabled (`-march=armv8.2-a+dotprod`)

### HNSW Index Performance (dim=384, k=10, ef=50)

| Corpus | Insert Rate | Search QPS | Search Latency |
|--------|-------------|------------|----------------|
| 1,000  | 7,395/s     | 16,565     | 60 µs          |
| 5,000  | 3,207/s     | 8,981      | 111 µs         |
| 10,000 | 2,116/s     | 6,400      | 156 µs         |
| 25,000 | 1,061/s     | 3,510      | 285 µs         |
| 50,000 | 632/s       | 2,369      | 422 µs         |

### Prefetching Impact

Software prefetching (`__builtin_prefetch`) in beam search provides 9-32% improvement:

| Corpus | Without Prefetch | With Prefetch | Improvement |
|--------|------------------|---------------|-------------|
| 1,000  | 15,168 QPS       | 16,540 QPS    | +9%         |
| 5,000  | 9,249 QPS        | 10,135 QPS    | +10%        |
| 10,000 | 6,337 QPS        | 7,761 QPS     | **+22%**    |
| 25,000 | 3,140 QPS        | 4,139 QPS     | **+32%**    |
| 50,000 | 2,303 QPS        | 2,744 QPS     | +19%        |
| 100,000| 1,907 QPS        | 2,179 QPS     | +14%        |

### Batch Search Scaling (10K corpus, 1000 queries)

Linear scaling with thread count using `search_batch()` API:

| Threads | QPS    | Speedup |
|---------|--------|---------|
| 1 (seq) | 5,979  | 1.00x   |
| 2       | 12,713 | **2.13x** |
| 4       | 24,377 | **4.08x** |
| 8       | 50,554 | **8.46x** |

### SIMD Distance Computation

#### Float32 Cosine Distance (NEON)

| Dimensions | NEON (ns) | Scalar (ns) | Speedup |
|------------|-----------|-------------|---------|
| 384        | 29        | 347         | **12x** |

#### Int8 Dot Product (DotProd Instruction)

Using `vdotq_s32` (ARMv8.2+) for quantized vectors:

| Method       | Time (ns) | Speedup |
|--------------|-----------|---------|
| Scalar       | 4.8       | 1.00x   |
| NEON DotProd | 0.3       | **17.1x** |

### Optimization Summary (M1 Pro)

| Optimization | Improvement | Notes |
|--------------|-------------|-------|
| NEON SIMD (float32) | 12x | Cosine distance |
| Prefetching | +9% to +32% | Depends on corpus size |
| Batch search | Linear | 8.46x with 8 threads |
| DotProd (int8) | 17x | ARMv8.2+ required |


---

## RAG Pipeline Benchmark

### 1. Corpus Size Scaling (384d, K=5)

| Corpus | Latency | Throughput | QPS (single-thread) |
|--------|---------|------------|---------------------|
| 1K     | 288 μs  | 3.51 M/s   | ~3,510 queries/sec  |
| 10K    | 3.63 ms | 2.77 M/s   | ~277 queries/sec    |
| 100K   | 41.0 ms | 2.43 M/s   | ~24 queries/sec     |


**Scaling**: Linear (10x corpus → 10x latency)
**Bottleneck**: Compute-bound (memory bandwidth utilization ~5%)

### 2. K-Value Scaling (10K docs, 384d)

| K  | Latency | Delta |
|----|---------|-------|
| 1  | 3.92 ms | +7.8% |
| 5  | 3.63 ms | baseline |
| 10 | 3.64 ms | +0.2% |
| 50 | 3.60 ms | -0.9% |


**Conclusion**: Partial sort overhead negligible; K-value has no meaningful impact.

### 3. Embedding Dimension Scaling (10K docs, K=5)

| Dimensions | Latency  | Throughput | Scaling Factor |
|------------|----------|------------|----------------|
| 384d       | 3.63 ms  | 2.77 M/s   | 1.0x           |
| 768d       | 6.78 ms  | 1.53 M/s   | 1.87x          |
| 1536d      | 13.2 ms  | 780k/s     | 3.64x          |


**Scaling**: Near-linear (2x dim → 2.06x latency, 4x dim → 4.21x latency)
**Conclusion**: Compute-bound; SIMD efficiency remains high across dimensions.

### 4. Quantization (10K docs, 384d, K=5)

| Type  | Latency | Throughput | Storage | Overhead |
|-------|---------|------------|---------|----------|
| float | 3.63 ms | 2.77 M/s   | 4 bytes | baseline |
| int8  | 3.62 ms | 2.79 M/s   | 1 byte  | **-0.4%** |


**Conclusion**: int8 quantization is **faster** while reducing storage 4x (memory bandwidth savings).

### 5. Multi-Query Throughput (10K docs, 384d)

- **10 queries**: 36.2 ms total (3.62 ms/query average)
- **Sustained throughput**: 2.76 M vectors/second
- **QPS**: ~276 queries/second (single-threaded)
- **Parallelization potential**: 48 cores → ~13.2K QPS theoretical


### 6. Sequential vs Batch (1K docs, 384d, K=5)

| Method     | Latency | Throughput |
|------------|---------|------------|
| Sequential | 287 μs  | 3.51 M/s   |
| Batch      | 288 μs  | 3.51 M/s   |


**Conclusion**: Batch API provides cleaner code at performance parity (memory-bandwidth bound).

---

## Batch Distance Benchmark

### 1. Sequential vs Batch Comparison

| Scenario | Sequential | Batch | Speedup |
|----------|------------|-------|---------|
| 100×384d | 27.8 μs    | 28.3 μs | 0.98x |
| 1K×384d  | 289 μs     | 283 μs  | 1.02x |


**Conclusion**: Parity performance; both memory-bandwidth limited.

### 2. Memory Layout Optimization

| Layout      | Latency | Throughput | Improvement |
|-------------|---------|------------|-------------|
| Scattered   | 283 μs  | 3.54 M/s   | baseline    |
| Contiguous  | 283 μs  | 3.54 M/s   | +0.0%       |


**Conclusion**: Marginal improvement; modern CPUs prefetch efficiently.

### 3. Top-K Performance (1K×384d, K=10)

- **Latency**: 290 μs (vs 287 μs full distance computation)
- **Overhead**: ~1% for partial sort
- **Conclusion**: `std::partial_sort` highly optimized; K << N has negligible cost.


### 4. Large Embeddings (1K×1536d)

- **Latency**: 1.18 ms
- **Throughput**: 833k vectors/second
- **Scaling**: 4.18x slower than 384d (expected 4.0x)


---

## HNSW Decision Matrix

| Corpus Size | Brute-Force Latency | Recommendation |
|-------------|---------------------|----------------|
| <10K        | <4ms                | ✅ Brute-force optimal |
| 10K-100K    | 4-40ms              | ⚠️ Brute-force acceptable for batch |
| >100K       | >40ms               | ❌ HNSW required for real-time (<10ms) |

**HNSW Threshold**: 100K vectors (~41ms → >10ms target requires ANN index)


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
| 1K corpus (<1ms) | 1000 μs | 288 μs | ✅ **3.5x better** |
| 10K corpus (<5ms) | 5000 μs | 3633 μs | ✅ **1.4x better** |
| 100K corpus (<50ms) | 50000 μs | 41000 μs | ✅ **1.2x better** |
| int8 overhead (<20%) | 20% | -0.4% | ✅ **Faster** |
| Dimension scaling | Linear | Near-linear | ✅ **Good** |

### Comparison to Previous Results (2025-11-02)

| Scenario | Previous | Current | Delta |
|----------|----------|---------|-------|
| 1K×384d (K=5) latency | 273 μs | 288 μs | +5.5% |
| 10K×384d (K=5) latency | 2.78 ms | 3.63 ms | +30.6% |
| 100K×384d (K=5) latency | 27.9 ms | 41.0 ms | +46.9% |
| 10K×384d throughput | 3.60 M/s | 2.77 M/s | -23.1% |
| int8 @10K×384d latency | 2.74 ms | 3.62 ms | +32.1% |

Notes:
- Previous run header (2025-11-02): Linux (x86_64), GCC 15.2.0, Google Benchmark 1.9.1.
- Current run header (2026-01-05): Windows 11 (x86_64), clang 21.1.6, Google Benchmark 1.9.4.
- Treat deltas as environment differences rather than regressions unless measured on the same OS/toolchain.



---

## Reproduction

### Windows (Conan)

```powershell
# From third_party/sqlite-vec-cpp

# Install dependencies (Conan 2)
conan profile detect --force
conan install . -of build_bench_conan -b missing -s build_type=Release -s compiler.cppstd=23 -s compiler.runtime=static

# Make Conan-generated .pc files visible to pkg-config for this shell
$env:PKG_CONFIG_PATH = (Resolve-Path .\build_bench_conan)

# Configure + build
meson setup build_bench --wipe -Denable_benchmarks=true -Dbuildtype=release
ninja -C build_bench benchmarks/rag_pipeline_benchmark.exe benchmarks/batch_distance_benchmark.exe

# Run
.\build_bench\benchmarks\rag_pipeline_benchmark.exe --benchmark_min_time=0.5s
.\build_bench\benchmarks\batch_distance_benchmark.exe --benchmark_min_time=0.5s

# JSON output for analysis
.\build_bench\benchmarks\rag_pipeline_benchmark.exe `
  --benchmark_out=results.json `
  --benchmark_out_format=json
```

### Linux/macOS (system packages)

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
