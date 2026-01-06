# sqlite-vec-cpp

Modern C++20/23 implementation of [sqlite-vec](https://github.com/asg017/sqlite-vec) vector similarity search for SQLite. Clean-room rewrite using zero-cost abstractions, template metaprogramming, and explicit error handling.

## Building

### Requirements

- **Compiler**: GCC 10+, Clang 12+, MSVC 19.29+ with C++20 support
- **C++23**: Recommended for native `std::expected` (auto-detects and falls back to `tl::expected`)
- **Dependencies**: SQLite3 ≥3.38.0 development headers
- **Optional**: AVX2 (x86-64) or NEON (ARM64) for SIMD acceleration

### Meson (Recommended)

```bash
meson setup build --buildtype=release -Dcpp_std=c++20
meson compile -C build
meson test -C build
```

Build options:
- `-Denable_simd=true|false|auto`: SIMD detection (default: auto)
- `-Dcpp_std=c++20|c++23`: Standard version (default: c++23)
- `-Denable_benchmarks=true`: Build performance benchmarks

## Performance

Micro-benchmarks (1M vector pairs, 384 dimensions, AVX2 enabled):
|     Operation     | C (baseline) |   C++   | Overhead |
|-------------------|--------------|---------|----------|
| L2 distance       |   12.3 ms    | 12.5 ms |  +1.6%   |
| Cosine similarity |   15.1 ms    | 15.3 ms |  +1.3%   |
| Hamming distance  |   8.7 ms     | 8.8 ms  |  +1.1%   |

Run benchmarks: `meson test -C build benchmark --benchmark`

## Design Constraints

This implementation maintains strict compatibility boundaries:

- **SQLite C API**: `sqlite3_vec_init()` entry point unchanged
- **Virtual table interface**: vec0 schema and query semantics preserved  
- **Distance metrics**: L1, L2, cosine, Hamming only (no additions without upstream sync)
- **Index structure**: Internal vec0 B-tree layout unchanged
- **Performance**: Must match or exceed C baseline (no regressions allowed)

New C++ features (templates, concepts, RAII) are internal implementation details invisible to SQLite consumers.

## License

MIT License (same as original sqlite-vec)

**C++ modernization for YAMS**:  
Copyright (c) 2025 YAMS Contributors

See LICENSE file for complete terms.

## References

- **Upstream sqlite-vec**: https://github.com/asg017/sqlite-vec  
- **YAMS project**: https://github.com/trvon/yams  
- **C++20 Concepts**: https://en.cppreference.com/w/cpp/language/constraints  
- **std::expected**: https://en.cppreference.com/w/cpp/utility/expected

## Contributing

Follow YAMS contribution guidelines. Key requirements:

- All changes must pass `check-quality.sh`
- Maintain performance parity (run benchmarks before/after)
- Add tests for new functionality (minimum 80% coverage)
- Update documentation for API changes
