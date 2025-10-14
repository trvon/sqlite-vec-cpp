# Benchmark suite for sqlite-vec-cpp
# Requires Google Benchmark library

benchmark_dep = dependency('benchmark', required: false)

if benchmark_dep.found()
  distance_benchmark_exe = executable(
    'distance_benchmark',
    'distance_benchmark.cpp',
    dependencies: [benchmark_dep],
    include_directories: include_directories('../include'),
    cpp_args: cpp_args,
    install: false
  )
  
  message('Benchmarks enabled - run with: ./distance_benchmark')
else
  message('Google Benchmark not found - benchmarks disabled')
  message('Install with: apt install libbenchmark-dev or brew install google-benchmark')
endif
