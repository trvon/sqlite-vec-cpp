project('sqlite-vec-cpp', 'cpp',
  version: '0.1.0',
  default_options: [
    'cpp_std=c++20',
    'warning_level=3',
    'werror=false',
    'buildtype=release',
  ],
  meson_version: '>=0.63.0'
)

# Import filesystem module
fs = import('fs')

# C++23 check (for std::expected)
cpp = meson.get_compiler('cpp')
has_cpp23 = cpp.has_argument('-std=c++23')

# =============================================================================
# SIMD Capability Detection
# =============================================================================
# Detect CPU architecture and available SIMD instructions at compile time.
# Users can override with -Denable_simd_avx=false or -Denable_simd_neon=false

host_cpu = host_machine.cpu_family()
simd_args = []
simd_defines = []

# ARM NEON Detection (AArch64 has NEON by default)
has_neon = false
has_arm_dotprod = false

if host_cpu == 'aarch64'
  if get_option('enable_simd_neon')
    # AArch64 always has NEON
    has_neon = true
    simd_defines += '-DSQLITE_VEC_ENABLE_NEON'
    message('NEON: enabled (AArch64 default)')
    
    # Check for ARMv8.2+ DotProd instruction (vdotq_s32)
    # Available on Apple M1+, Cortex-A75+, Neoverse N1+
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
    # Try with +dotprod arch extension
    if cpp.compiles(dotprod_test, args: ['-march=armv8.2-a+dotprod'], name: 'ARM DotProd (vdotq_s32)')
      has_arm_dotprod = true
      simd_args += '-march=armv8.2-a+dotprod'
      message('ARM DotProd: enabled (ARMv8.2+)')
    else
      message('ARM DotProd: not available')
    endif
  else
    message('NEON: disabled by user')
  endif
elif host_cpu == 'arm'
  # 32-bit ARM may or may not have NEON
  if get_option('enable_simd_neon')
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
      message('NEON: enabled (32-bit ARM)')
    else
      message('NEON: not available')
    endif
  else
    message('NEON: disabled by user')
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
      message('AVX: enabled')
      
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
        message('AVX2: enabled')
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
        message('FMA: enabled')
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
        message('AVX-512: enabled')
      else
        message('AVX-512: not available')
      endif
    else
      message('AVX: not available')
    endif
  else
    message('AVX: disabled by user')
  endif
endif

# Apply SIMD flags globally
if simd_args.length() > 0
  add_project_arguments(simd_args, language: 'cpp')
endif
if simd_defines.length() > 0
  add_project_arguments(simd_defines, language: 'cpp')
endif

# Check if std::expected is actually available
expected_test = '''
#include <expected>
int main() {
    std::expected<int, int> e = 42;
    return e.has_value() ? 0 : 1;
}
'''
has_std_expected = cpp.compiles(expected_test, 
  args: ['-std=c++23'], 
  name: 'std::expected availability')

if has_std_expected
  message('std::expected available, using C++23 std::expected')
  add_project_arguments('-DHAS_CPP23_EXPECTED', language: 'cpp')
  add_project_arguments('-std=c++23', language: 'cpp')
else
  message('std::expected not available, using fallback Expected<T,E>')
endif

# SQLite3 dependency
# Try multiple methods to find SQLite3:
# 1. pkg-config (Unix-y systems)
# 2. cmake (Windows with vcpkg, some Unix systems)
# 3. system (fallback)
sqlite3_dep = dependency('sqlite3', version: '>=3.38.0', required: false, method: 'pkg-config')
if not sqlite3_dep.found()
  sqlite3_dep = dependency('sqlite3', version: '>=3.38.0', required: false, method: 'cmake')
endif
if not sqlite3_dep.found()
  sqlite3_dep = dependency('sqlite3', version: '>=3.38.0', required: true)
endif

# Project include directories
inc_dirs = include_directories('include')

# Public headers (for installation)
public_headers = [
  'include/sqlite-vec-cpp/sqlite_vec.hpp',
  'include/sqlite-vec-cpp/vector_view.hpp',
]

# Subdirectories
subdir('include/sqlite-vec-cpp/concepts')
subdir('include/sqlite-vec-cpp/distances')
subdir('include/sqlite-vec-cpp/utils')
subdir('src')

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
  'C++ Standard': has_cpp23 ? 'C++23' : 'C++20',
  'SQLite Version': sqlite3_dep.version(),
  'Build Type': get_option('buildtype'),
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
