# Benchmark suite for sqlite-vec-cpp
# Requires Google Benchmark library

benchmark_dep = dependency('benchmark', required: false)

# Optional cpp_args (only used when building within YAMS)
cpp_args_list = []
if meson.is_subproject()
  cpp_args_list = cpp_args
endif

if benchmark_dep.found()
  distance_benchmark_exe = executable(
    'distance_benchmark',
    'distance_benchmark.cpp',
    dependencies: [benchmark_dep],
    include_directories: include_directories('../include'),
    cpp_args: cpp_args_list,
    install: false
  )

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
  message('  ./distance_benchmark')
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

# Quantized search benchmark (standalone, no Google Benchmark dependency)
quantized_search_benchmark_exe = executable(
  'quantized_search_benchmark',
  'quantized_search_benchmark.cpp',
  include_directories: include_directories('../include'),
  cpp_args: cpp_args_list,
  install: false
)
message('  ./quantized_search_benchmark (LVQ-8, LVQ-4, RaBitQ vs FP32 baseline)')

# HNSW M-parameter sweep benchmark (standalone, no Google Benchmark dependency)
# Systematically tests M values from 8 to 32 to identify the recall/cost knee.
hnsw_m_sweep_benchmark_exe = executable(
  'hnsw_m_sweep_benchmark',
  'hnsw_m_sweep_benchmark.cpp',
  include_directories: include_directories('../include'),
  cpp_args: cpp_args_list,
  install: false
)
message('  ./hnsw_m_sweep_benchmark (M=8..32 recall/cost Pareto sweep)')

# HNSW ef_construction sweep benchmark (standalone, no Google Benchmark dependency)
# Tests ef_construction 50..500 to validate for_corpus() thresholds.
hnsw_ef_sweep_benchmark_exe = executable(
  'hnsw_ef_sweep_benchmark',
  'hnsw_ef_sweep_benchmark.cpp',
  include_directories: include_directories('../include'),
  cpp_args: cpp_args_list,
  install: false
)
message('  ./hnsw_ef_sweep_benchmark (ef_construction 50..500 sweep)')

vec0_ann_benchmark_exe = executable(
  'vec0_ann_benchmark',
  'vec0_ann_benchmark.cpp',
  include_directories: include_directories('../include'),
  sources: ['../src/sqlite_vec_c_api.cpp'],
  dependencies: [sqlite3_dep],
  cpp_args: cpp_args_list,
  install: false
)
message('  ./vec0_ann_benchmark (warm vec0 ANN vs exact SQL full scan)')

profile_vec0_query_exe = executable(
  'profile_vec0_query',
  'profile_vec0_query.cpp',
  include_directories: include_directories('../include'),
  sources: ['../src/sqlite_vec_c_api.cpp'],
  dependencies: [sqlite3_dep],
  cpp_args: cpp_args_list,
  install: false
)
message('  ./profile_vec0_query (cold/hot vec0 ANN profiling with SQLite statement timings)')
