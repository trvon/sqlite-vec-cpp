project('sqlite-vec-cpp', 'cpp',
  version: '0.1.0',
  default_options: [
    'cpp_std=c++20',
    'warning_level=3',
    'werror=false',
    'buildtype=debugoptimized',
    'b_ndebug=if-release',
  ],
  meson_version: '>=0.63.0'
)

# Import filesystem module
fs = import('fs')
pkg = import('pkgconfig')

cpp = meson.get_compiler('cpp')
buildtype = get_option('buildtype')
build_cpp_std = get_option('cpp_std')

# Meson's debugoptimized mode does not define NDEBUG by default. Add it here so
# the default build behaves like RelWithDebInfo instead of a hybrid debug build.
if buildtype == 'debugoptimized'
  add_project_arguments('-DNDEBUG', language: 'cpp')
endif

# =============================================================================
# SIMD Capability Detection
# =============================================================================
# Detect CPU architecture and available SIMD instructions at compile time.
# Users can override with -Denable_simd_avx=false or -Denable_simd_neon=false

host_cpu = host_machine.cpu_family()
simd_args = []
simd_defines = []

if meson.is_cross_build() and (get_option('enable_simd_avx') or get_option('enable_simd_arm32_neon') or get_option('enable_simd_dotprod'))
  warning('Cross-build with CPU-specific SIMD tuning enabled. Ensure these options match the target CPU; portable defaults leave them disabled.')
endif

# ARM NEON Detection (AArch64 has NEON by default)
has_neon = false
has_arm_dotprod = false

if host_cpu == 'aarch64'
  if get_option('enable_simd_neon')
    # AArch64 always has NEON
    has_neon = true
    simd_defines += '-DSQLITE_VEC_ENABLE_NEON'
    message('NEON: enabled (AArch64 baseline)')

    if get_option('enable_simd_dotprod')
      # Check for ARMv8.2+ DotProd instruction (vdotq_s32).
      dotprod_test = '''
      #include <arm_neon.h>
      int main() {
          int8x16_t a = vdupq_n_s8(1);
          int8x16_t b = vdupq_n_s8(2);
          int32x4_t c = vdupq_n_s32(0);
          c = vdotq_s32(c, a, b);
          return vgetq_lane_s32(c, 0);
      }
      '''
      if cpp.compiles(dotprod_test, args: ['-march=armv8.2-a+dotprod'], name: 'ARM DotProd (vdotq_s32)')
        has_arm_dotprod = true
        simd_args += '-march=armv8.2-a+dotprod'
        message('ARM DotProd: enabled (non-portable opt-in)')
      else
        message('ARM DotProd: requested but not available for this toolchain')
      endif
    else
      message('ARM DotProd: disabled by default (enable_simd_dotprod=false)')
    endif
  else
    message('NEON: disabled by user')
  endif
elif host_cpu == 'arm'
  # 32-bit ARM NEON requires a non-portable compiler flag, so keep it opt-in.
  if get_option('enable_simd_arm32_neon')
    neon_test = '''
    #include <arm_neon.h>
    int main() {
        float32x4_t v = vdupq_n_f32(1.0f);
        return 0;
    }
    '''
    if cpp.compiles(neon_test, args: ['-mfpu=neon'], name: 'ARM NEON (32-bit)')
      has_neon = true
      simd_defines += '-DSQLITE_VEC_ENABLE_NEON'
      simd_args += '-mfpu=neon'
      message('NEON: enabled for 32-bit ARM (non-portable opt-in)')
    else
      message('NEON: requested for 32-bit ARM but not available for this toolchain')
    endif
  else
    message('NEON: disabled for 32-bit ARM by default (enable_simd_arm32_neon=false)')
  endif
endif

# x86/x86_64 AVX Detection
has_avx = false
has_avx2 = false
has_fma = false
has_avx512 = false

if host_cpu in ['x86', 'x86_64']
  if get_option('enable_simd_avx')
    # Test AVX
    avx_test = '''
    #include <immintrin.h>
    int main() {
        __m256 v = _mm256_set1_ps(1.0f);
        __m256 r = _mm256_add_ps(v, v);
        return 0;
    }
    '''
    if cpp.compiles(avx_test, args: ['-mavx'], name: 'AVX')
      has_avx = true
      simd_defines += '-DSQLITE_VEC_ENABLE_AVX'
      simd_args += '-mavx'
      message('AVX: enabled (non-portable opt-in)')
      
      # Test AVX2 (needed for integer operations)
      avx2_test = '''
      #include <immintrin.h>
      int main() {
          __m256i v = _mm256_set1_epi32(1);
          __m256i r = _mm256_add_epi32(v, v);
          return 0;
      }
      '''
      if cpp.compiles(avx2_test, args: ['-mavx2'], name: 'AVX2')
        has_avx2 = true
        simd_defines += '-DSQLITE_VEC_ENABLE_AVX2'
        simd_args += '-mavx2'
        message('AVX2: enabled (non-portable opt-in)')
      else
        message('AVX2: not available')
      endif
      
      # Test FMA (fused multiply-add, usually with AVX2)
      fma_test = '''
      #include <immintrin.h>
      int main() {
          __m256 a = _mm256_set1_ps(1.0f);
          __m256 b = _mm256_set1_ps(2.0f);
          __m256 c = _mm256_set1_ps(3.0f);
          __m256 r = _mm256_fmadd_ps(a, b, c);
          return 0;
      }
      '''
      if cpp.compiles(fma_test, args: ['-mfma'], name: 'FMA')
        has_fma = true
        simd_defines += '-DSQLITE_VEC_ENABLE_FMA'
        simd_args += '-mfma'
        message('FMA: enabled (non-portable opt-in)')
      else
        message('FMA: not available')
      endif
      
      # Test AVX-512 (optional, for future use)
      avx512_test = '''
      #include <immintrin.h>
      int main() {
          __m512 v = _mm512_set1_ps(1.0f);
          __m512 r = _mm512_add_ps(v, v);
          return 0;
      }
      '''
      if cpp.compiles(avx512_test, args: ['-mavx512f'], name: 'AVX-512')
        has_avx512 = true
        simd_defines += '-DSQLITE_VEC_ENABLE_AVX512'
        simd_args += '-mavx512f'
        message('AVX-512: enabled (non-portable opt-in)')
      else
        message('AVX-512: not available')
      endif
    else
      message('AVX: requested but not available for this toolchain')
    endif
  else
    message('AVX: disabled by default (enable_simd_avx=false)')
  endif
endif

# Apply SIMD flags globally
if simd_args.length() > 0
  add_project_arguments(simd_args, language: 'cpp')
endif
if simd_defines.length() > 0
  add_project_arguments(simd_defines, language: 'cpp')
endif

# Check if std::expected is actually available with the configured C++ standard.
expected_test = '''
#include <expected>
int main() {
    std::expected<int, int> e = 42;
    return e.has_value() ? 0 : 1;
}
'''
has_std_expected = cpp.compiles(expected_test, name: 'std::expected availability')

if get_option('use_system_expected')
  warning('use_system_expected is not implemented yet; using std::expected when available, otherwise the built-in fallback Expected<T, E>.')
endif

if has_std_expected
  message('std::expected available with ' + build_cpp_std)
  add_project_arguments('-DHAS_CPP23_EXPECTED', language: 'cpp')
else
  message('std::expected not available with ' + build_cpp_std + ', using fallback Expected<T, E>')
endif

# SQLite3 dependency
# Try multiple methods to find SQLite3:
# 1. pkg-config (Unix-y systems, MinGW)
# 2. cmake with unofficial-sqlite3 (vcpkg on Windows)
# 3. cmake with sqlite3 (some systems)
# 4. system (fallback - lets meson try everything)
sqlite3_dep = dependency('sqlite3', version: '>=3.38.0', required: false, method: 'pkg-config')
if not sqlite3_dep.found()
  # vcpkg uses 'unofficial-sqlite3' as the cmake package name
  sqlite3_dep = dependency('unofficial-sqlite3', required: false, method: 'cmake', modules: ['unofficial::sqlite3::sqlite3'])
endif
if not sqlite3_dep.found()
  sqlite3_dep = dependency('sqlite3', version: '>=3.38.0', required: false, method: 'cmake')
endif
if not sqlite3_dep.found()
  # Final fallback - let meson try all methods
  sqlite3_dep = dependency('sqlite3', version: '>=3.38.0', required: true)
endif

# Project include directories
inc_dirs = include_directories('include')

# Subdirectories
subdir('include/sqlite-vec-cpp/concepts')
subdir('include/sqlite-vec-cpp/distances')
subdir('include/sqlite-vec-cpp/utils')
subdir('src')

install_headers(
  [
    'include/sqlite-vec-cpp/sqlite_vec.hpp',
    'include/sqlite-vec-cpp/vector_view.hpp',
  ],
  subdir: 'sqlite-vec-cpp',
)

install_headers(
  [
    'include/sqlite-vec-cpp/distances/batch.hpp',
    'include/sqlite-vec-cpp/distances/cosine.hpp',
    'include/sqlite-vec-cpp/distances/hamming.hpp',
    'include/sqlite-vec-cpp/distances/inner_product.hpp',
    'include/sqlite-vec-cpp/distances/l1.hpp',
    'include/sqlite-vec-cpp/distances/l2.hpp',
  ],
  subdir: 'sqlite-vec-cpp/distances',
)

install_headers(
  [
    'include/sqlite-vec-cpp/index/hnsw.hpp',
    'include/sqlite-vec-cpp/index/hnsw_node.hpp',
    'include/sqlite-vec-cpp/index/hnsw_persistence.hpp',
    'include/sqlite-vec-cpp/index/hnsw_quantized.hpp',
    'include/sqlite-vec-cpp/index/hnsw_threading.hpp',
    'include/sqlite-vec-cpp/index/quantization_snapshot.hpp',
  ],
  subdir: 'sqlite-vec-cpp/index',
)

install_headers(
  [
    'include/sqlite-vec-cpp/simd/avx.hpp',
    'include/sqlite-vec-cpp/simd/neon.hpp',
  ],
  subdir: 'sqlite-vec-cpp/simd',
)

install_headers(
  [
    'include/sqlite-vec-cpp/sqlite/column_defs.hpp',
    'include/sqlite-vec-cpp/sqlite/context.hpp',
    'include/sqlite-vec-cpp/sqlite/enhanced_functions.hpp',
    'include/sqlite-vec-cpp/sqlite/functions.hpp',
    'include/sqlite-vec-cpp/sqlite/parsers.hpp',
    'include/sqlite-vec-cpp/sqlite/registration.hpp',
    'include/sqlite-vec-cpp/sqlite/utility_functions.hpp',
    'include/sqlite-vec-cpp/sqlite/value.hpp',
    'include/sqlite-vec-cpp/sqlite/vec0_module.hpp',
    'include/sqlite-vec-cpp/sqlite/vtab.hpp',
  ],
  subdir: 'sqlite-vec-cpp/sqlite',
)

install_headers(
  [
    'include/sqlite-vec-cpp/utils/array.hpp',
    'include/sqlite-vec-cpp/utils/float16.hpp',
  ],
  subdir: 'sqlite-vec-cpp/utils',
)

install_headers(
  [
    'include/sqlite-vec-cpp/quantization/lvq.hpp',
    'include/sqlite-vec-cpp/quantization/rabitq.hpp',
    'include/sqlite-vec-cpp/quantization/store.hpp',
  ],
  subdir: 'sqlite-vec-cpp/quantization',
)


pkg.generate(
  sqlite_vec_cpp_lib,
  filebase: 'sqlite-vec-cpp',
  name: 'sqlite-vec-cpp',
  description: 'Modern C++ wrapper and compatibility layer for sqlite-vec',
  version: meson.project_version(),
  requires: ['sqlite3 >= 3.38.0'],
  subdirs: '.',
  extra_cflags: ['-DSQLITE_VEC_CPP_ENABLED'],
)

cmake_pkg = configuration_data()
cmake_pkg.set('VERSION', meson.project_version())
cmake_pkg.set('LIBDIR', get_option('libdir'))
cmake_pkg.set('INCLUDEDIR', get_option('includedir'))

sqlite_vec_cpp_cmake_config = configure_file(
  input: files('packaging/sqlite-vec-cppConfig.cmake.in'),
  output: 'sqlite-vec-cppConfig.cmake',
  configuration: cmake_pkg,
)

sqlite_vec_cpp_cmake_version = configure_file(
  input: files('packaging/sqlite-vec-cppConfigVersion.cmake.in'),
  output: 'sqlite-vec-cppConfigVersion.cmake',
  configuration: cmake_pkg,
)

install_data(
  [sqlite_vec_cpp_cmake_config, sqlite_vec_cpp_cmake_version],
  install_dir: join_paths(get_option('libdir'), 'cmake', 'sqlite-vec-cpp'),
)

# Optional: tests
if get_option('enable_tests')
  subdir('tests')
endif

# Optional: examples
if get_option('enable_examples')
  subdir('examples')
endif

# Optional: benchmarks
if get_option('enable_benchmarks')
  subdir('benchmarks')
endif

# Summary
summary({
  'C++ Standard': build_cpp_std,
  'std::expected': has_std_expected,
  'SQLite Version': sqlite3_dep.version(),
  'Build Type': buildtype,
  'Tests': get_option('enable_tests'),
  'Examples': get_option('enable_examples'),
  'Benchmarks': get_option('enable_benchmarks'),
}, section: 'Configuration')

# SIMD Summary
simd_summary = {}
if host_cpu == 'aarch64' or host_cpu == 'arm'
  simd_summary += {'NEON': has_neon}
  simd_summary += {'ARM DotProd (vdotq_s32)': has_arm_dotprod}
elif host_cpu in ['x86', 'x86_64']
  simd_summary += {'AVX': has_avx}
  simd_summary += {'AVX2': has_avx2}
  simd_summary += {'FMA': has_fma}
  simd_summary += {'AVX-512': has_avx512}
endif
summary(simd_summary, section: 'SIMD Features')
