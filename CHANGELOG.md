# Changelog

All notable changes to sqlite-vec-cpp will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2026-06-09

### Fixed
- **Correctness audit (Phase A)**:
  - HNSW persistence: cap `num_layers` (`kMaxHnswLayers = 64`) and per-layer
    neighbor counts during node deserialization — corrupt shadow-table blobs
    previously triggered unbounded allocations (OOM). Deleted-ids blobs now
    validate their count against the blob size before `reserve`.
  - `load_hnsw_index`: validate node-id consistency between the shadow-table
    rowid and the serialized blob, reject a missing entry point, and prune
    dangling neighbor edges referencing nodes absent from the shadow table.
  - `get_hnsw_checkpoint_info`: bounds-check the entry-point meta blob before
    dereferencing (previously an unchecked cast + read).
  - vec0: `dimensions * sizeof(float)` size checks now use `size_t` arithmetic
    (previously truncated through `int`); `CREATE VIRTUAL TABLE` rejects
    dimensions outside `[1, 65536]` and embedding column names containing `"`.
  - vec0 xFilter: ANN plans without an explicit `k` constraint no longer
    dereference a disengaged `std::optional` (use `kVec0DefaultK`).
  - HNSW greedy descent: NaN distances no longer "improve" the current
    candidate — NaN/Inf vectors previously caused an infinite traversal loop
    on insert and search (found by the new adversarial test).
  - HNSW concurrent search: `nodes_.empty()` was read before acquiring
    `nodes_mutex_` in `search_with_filter_impl` / `search_quantized_rerank` /
    `search_batch_with_filter` — a data race against concurrent `insert()`
    (found by the new TSan concurrency stress test).
  - `insert_single_threaded`: debug-build guard asserts on concurrent callers.

### Added
- **True RaBitQ quantization** (arXiv:2405.12497), replacing the previous
  sign-quantization heuristic: seeded FWHT-based random rotation, the paper's
  unbiased inner-product estimator with per-vector correction factors, and
  4-bit scalar quantization of the rotated query so distance estimation stays
  popcount-only (5 popcounts per candidate). Recall@10 on 10K Gaussian
  vectors (3x rerank, ef=100): 76.6% → 94.9% at 128d, and at 768d RaBitQ now
  dominates the LVQ latency/recall/memory frontier (99.6% @ ef=200/3x, ~2.9x
  less memory than LVQ-4, faster than FP32 traversal). See BENCHMARKS.md.
- **Hardening tests**: deterministic persistence fuzzing (truncations, byte
  flips, hostile length splices), adversarial vector tests (NaN/Inf/zero/
  denormal through distances, HNSW, LVQ/RaBitQ), vec0 overflow/validation
  tests, and a concurrent insert+search stress test (TSan-clean).
- **hnsw_churn_benchmark**: recall/latency under sustained insert+delete
  churn per delete policy (soft / isolate_deleted / compact). Verdict: soft
  deletion exceeds fresh-built recall through 50% corpus turnover, so
  MN-RU-style repair-on-delete (arXiv:2407.07871) was evaluated and rejected
  — it only addresses degradation introduced by isolate_deleted(). See
  BENCHMARKS.md.

## [0.1.0] - 2025-11-02

### Added
- **C++20/23 Modernization**: Complete rewrite of sqlite-vec in modern C++
  - Template-based distance metrics with concept constraints
  - `std::span` for zero-copy vector operations
  - `std::expected` for type-safe error handling (C++23)
  - RAII wrappers for SQLite C API (Context, Value, VTab)
- **Distance Metrics**: L2 (Euclidean), L1 (Manhattan), Cosine, Hamming
  - Full template support for `float`, `int8_t`, `int16_t`
  - Conditional SIMD: AVX (x86_64), NEON (ARM64)
  - Zero-cost abstractions validated via benchmarks
- **Batch Operations** (Phase 2):
  - `batch_distance()`: 1 query vs N database vectors
  - `batch_distance_contiguous()`: Optimized for contiguous memory layout
  - `batch_top_k()`: Efficient top-K nearest neighbor search
  - `batch_distance_filtered()`: Distance threshold filtering
  - `batch_distance_parallel()`: C++17 parallel algorithms (optional)
- **vec0 Virtual Table Module**: Complete SQLite virtual table implementation
  - Shadow tables for metadata and row IDs
  - Full CRUD operations with type-safe C++ API
  - Integration with YAMS vector backend
- **Comprehensive Testing**:
  - 22 unit tests (Concepts, Distance Metrics, Utils, SQLite Functions, Batch Ops)
  - 100% pass rate, <0.1s execution time
- **Benchmarking Suite**:
  - RAG Pipeline Benchmark: 13 scenarios (corpus size, K-value, dimensions, quantization)
  - Batch Distance Benchmark: 8 scenarios (sequential vs batch, contiguous, int8)
  - Google Benchmark integration with JSON output

### Performance
- **Sub-microsecond latency**: 273 μs for 1K vectors (384d), 2.78 ms for 10K vectors
- **Sustained throughput**: 3.6M vectors/second across all corpus sizes
- **Linear scaling**: 2x dimensions → 2x latency (compute-bound)
- **int8 quantization**: 4x storage reduction at parity performance (1% faster)
- **K-value independence**: Top-K search overhead negligible (< 1%)

### Changed
- **Build System**: Meson with C++20/23 auto-detection
- **API Surface**: Replaced raw pointers with `std::span` throughout
- **Error Handling**: Migrated from C error codes to `std::expected<T, E>`

---

## [0.2.0] - 2025-11-02

### Added
- **HNSW Index (Phase 1 - Core Implementation)** (Task 057-109):
  - Header-only HNSW implementation with full C++20/23 support
  - Hierarchical graph structure with exponential layer assignment
  - Greedy search (upper layers) + Beam search with priority queues (layer 0)
  - Bidirectional edge connections with M_max pruning
  - Batch build support and configurable parameters (M, ef_construction)
  - **Files**: `hnsw.hpp` (327 lines), `hnsw_node.hpp` (61 lines)
- **HNSW Persistence Layer** (Partial):
  - Serialization/deserialization for config and nodes
  - Shadow table schema design (`_hnsw_meta`, `_hnsw_nodes`)
  - Save function (90% complete)
  - **File**: `hnsw_persistence.hpp` (310+ lines)
- **HNSW Benchmark Suite**:
  - Build time scaling (1K, 10K, 100K vectors)
  - Search latency vs corpus size
  - ef_search tuning (recall vs latency trade-off)
  - Brute-force comparison for speedup validation
  - **File**: `hnsw_benchmark.cpp` (290 lines)

### Performance (HNSW)
- **Recall quality**: 90-100% with ef_search=100-200 (10K vectors)
- **Graph connectivity**: 100% of nodes reachable from entry point
- **Build throughput**: ~1.6K vectors/sec (1K corpus), ~370 vectors/sec (10K corpus)
- **Search latency**: ~735 μs @ 10K vectors (ef=50)
- **Expected speedup** (vs brute-force):
  - 10K vectors: ~2x (1.5ms vs 2.8ms)
  - 100K vectors: ~14x (2ms vs 27.9ms)
  - 1M vectors: ~56x (5ms vs 280ms estimated)
- **Memory overhead**: ~80 bytes/vector (M=16, avg 3 layers)

### Known Limitations
- **SQLite Integration**: Incomplete (60% done) - deserialization, query planner, incremental updates pending

---
