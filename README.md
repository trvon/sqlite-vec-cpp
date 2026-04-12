# sqlite-vec-cpp

Modern C++20/23 implementation of [sqlite-vec](https://github.com/asg017/sqlite-vec) vector similarity search for SQLite. Clean-room rewrite using zero-cost abstractions, template metaprogramming, and explicit error handling.

## Building

### Requirements

- **Compiler**: GCC 10+, Clang 12+, MSVC 19.29+ with C++20 support
- **C++23**: Optional. If the configured build standard provides `std::expected`, it will be used automatically.
- **Dependencies**: SQLite3 ≥3.38.0 development headers
- **Optional**: CPU-specific SIMD tuning for AVX-family or ARM DotProd

### Meson (Recommended)

```bash
meson setup build --buildtype=debugoptimized -Db_ndebug=true -Dcpp_std=c++20
meson compile -C build
meson test -C build
```

This is the recommended default for packaging and distribution: portable optimized code with debug info.

### Offline / System Dependencies

`sqlite-vec-cpp` does not require Conan for normal builds. The primary integration path is direct Meson with system dependencies.

```bash
PKG_CONFIG_PATH=/opt/sqlite/lib/pkgconfig \
meson setup build \
  --buildtype=debugoptimized \
  -Db_ndebug=true \
  --wrap-mode=nofallback \
  -Dcpp_std=c++20

meson compile -C build
```

If your environment prefers CMake as the source-build entrypoint:

```bash
cmake -S . -B build/cmake-bootstrap
cmake --build build/cmake-bootstrap
```

### Install / Packaging

```bash
meson setup build --buildtype=debugoptimized -Db_ndebug=true
meson compile -C build
meson install -C build
```

Installed package metadata includes:
- `sqlite-vec-cpp.pc`
- `sqlite-vec-cppConfig.cmake`

### Conan (Optional Dev Bootstrap)

Conan is optional and primarily intended for local development convenience:

```bash
conan install . -of build/conan -s build_type=Release
```

For benchmark dependencies as well:

```bash
conan install . -of build/conan -s build_type=Release -o with_benchmarks=True
```

Build options:
- `-Dcpp_std=c++20|c++23`: Standard version (default: `c++20`)
- `-Denable_simd_neon=true|false`: baseline AArch64 NEON support (default: `true`)
- `-Denable_simd_dotprod=true`: enable ARMv8.2 DotProd (`-march=armv8.2-a+dotprod`, non-portable)
- `-Denable_simd_avx=true`: enable x86 AVX/AVX2/FMA/AVX-512 probing (`-mavx*`, non-portable)
- `-Denable_simd_arm32_neon=true`: enable 32-bit ARM NEON via `-mfpu=neon` (non-portable)
- `-Denable_benchmarks=true`: Build performance benchmarks

## Performance

Micro-benchmarks (1M vector pairs, 384 dimensions, AVX2 enabled):
|     Operation     | C (baseline) |   C++   | Overhead |
|-------------------|--------------|---------|----------|
| L2 distance       |   12.3 ms    | 12.5 ms |  +1.6%   |
| Cosine similarity |   15.1 ms    | 15.3 ms |  +1.3%   |
| Hamming distance  |   8.7 ms     | 8.8 ms  |  +1.1%   |

Run benchmarks: `meson test -C build benchmark --benchmark`

Benchmark configurations may intentionally use non-portable CPU tuning flags. Keep those settings separate from package-manager or distribution builds.

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
