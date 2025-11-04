# Benchmark suite for sqlite-vec-cpp
# Requires Google Benchmark library

benchmark_dep = dependency('benchmark', required: false)

# Optional cpp_args (only used when building within YAMS)
cpp_args_list = []
if meson.is_subproject()
  cpp_args_list = cpp_args
endif

if benchmark_dep.found()
  # Note: distance_benchmark.cpp needs updating for C++ API
  # Commented out temporarily
  # distance_benchmark_exe = executable(
  #   'distance_benchmark',
  #   'distance_benchmark.cpp',
  #   dependencies: [benchmark_dep],
  #   include_directories: include_directories('../include'),
  #   cpp_args: cpp_args_list,
  #   install: false
  # )

  batch_distance_benchmark_exe = executable(
    'batch_distance_benchmark',
    'batch_distance_benchmark.cpp',
    dependencies: [benchmark_dep],
    include_directories: include_directories('../include'),
    cpp_args: cpp_args_list,
    install: false
  )

  rag_pipeline_benchmark_exe = executable(
    'rag_pipeline_benchmark',
    'rag_pipeline_benchmark.cpp',
    dependencies: [benchmark_dep],
    include_directories: include_directories('../include'),
    cpp_args: cpp_args_list,
    install: false
  )

  hnsw_benchmark_exe = executable(
    'hnsw_benchmark',
    'hnsw_benchmark.cpp',
    dependencies: [benchmark_dep],
    include_directories: include_directories('../include'),
    cpp_args: cpp_args_list,
    install: false
  )

  message('Benchmarks enabled:')
  # message('  ./distance_benchmark')  # TODO: Update for C++ API
  message('  ./batch_distance_benchmark')
  message('  ./rag_pipeline_benchmark')
  message('  ./hnsw_benchmark')
else
  message('Google Benchmark not found - benchmarks disabled')
  message('Install with: apt install libbenchmark-dev or brew install google-benchmark')
endif
