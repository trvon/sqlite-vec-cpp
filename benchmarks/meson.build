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

  filtered_search_benchmark_exe = executable(
    'filtered_search_benchmark',
    'filtered_search_benchmark.cpp',
    dependencies: [benchmark_dep],
    include_directories: include_directories('../include'),
    cpp_args: cpp_args_list,
    install: false
  )

  precision_benchmark_exe = executable(
    'precision_benchmark',
    'precision_benchmark.cpp',
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
  message('  ./filtered_search_benchmark')
  message('  ./precision_benchmark')
else
  message('Google Benchmark not found - benchmarks disabled')
  message('Install with: apt install libbenchmark-dev or brew install google-benchmark')
endif

# HNSW engine comparison benchmark (standalone, no Google Benchmark dependency)
# Links against zvec if available; otherwise runs YAMS HNSW baseline only.
hnsw_engine_bench_cpp_args = cpp_args_list
hnsw_engine_bench_incs = [include_directories('../include')]
zvec_root = get_option('zvec-root')
if zvec_root != ''
  hnsw_engine_bench_incs += include_directories(zvec_root / 'src' / 'include')
  hnsw_engine_bench_cpp_args += ['-DYAMS_HAS_ZVEC=1']
  message('hnsw engine comparison benchmark: zvec ENABLED from ' + zvec_root)
else
  message('hnsw engine comparison benchmark: zvec NOT available (use -Dzvec-root=/path/to/zvec to enable)')
endif

hnsw_engine_comparison_benchmark_exe = executable(
  'hnsw_engine_comparison_benchmark',
  'hnsw_engine_comparison_benchmark.cpp',
  include_directories: hnsw_engine_bench_incs,
  cpp_args: hnsw_engine_bench_cpp_args,
  install: false
)
message('  ./hnsw_engine_comparison_benchmark (YAMS HNSW baseline + optional zvec)')
