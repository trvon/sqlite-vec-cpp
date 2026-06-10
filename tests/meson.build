# Tests

test_cpp_args = []
if cpp.get_id() != 'msvc'
  test_cpp_args = [
    '-Wno-unused-but-set-variable',
    '-Wno-unused-parameter',
    '-Wno-unused-variable',
  ]
endif

test_concepts = executable(
  'test_concepts',
  'test_concepts.cpp',
  include_directories: inc_dirs,
  dependencies: [sqlite3_dep],
  cpp_args: test_cpp_args,
)

test_distances = executable(
  'test_distances',
  'test_distances.cpp',
  include_directories: inc_dirs,
  dependencies: [sqlite3_dep],
  cpp_args: test_cpp_args,
)

test_utils = executable(
  'test_utils',
  'test_utils.cpp',
  include_directories: inc_dirs,
  dependencies: [sqlite3_dep],
  cpp_args: test_cpp_args,
)

test_sqlite_functions = executable(
  'test_sqlite_functions',
  'test_sqlite_functions.cpp',
  include_directories: inc_dirs,
  sources: [
    'test_sqlite_functions.cpp',
    '../src/sqlite_vec_c_api.cpp',
  ],
  dependencies: [sqlite3_dep],
  cpp_args: test_cpp_args + ['-DSQLITE_VEC_CPP_ENABLED', '-DSQLITE_VEC_CPP_TESTING'],
)

test_batch_distance = executable(
  'test_batch_distance',
  'test_batch_distance.cpp',
  include_directories: inc_dirs,
  dependencies: [sqlite3_dep],
  cpp_args: test_cpp_args,
)

test_hnsw = executable(
  'test_hnsw',
  'test_hnsw.cpp',
  include_directories: inc_dirs,
  dependencies: [sqlite3_dep],
  cpp_args: test_cpp_args,
)

test_tls_linkage = executable(
  'test_tls_linkage',
  [
    'test_tls_linkage.cpp',
    'test_tls_linkage_helper.cpp',
  ],
  include_directories: inc_dirs,
  dependencies: [sqlite3_dep],
  cpp_args: test_cpp_args,
)

# Detect if sanitizers are enabled (TSAN/ASAN add significant overhead)
# Meson doesn't expose b_sanitize directly, so we check compiler args
is_sanitized = get_option('b_sanitize') != 'none'
# Headroom matters: the TSAN suite has been observed at ~690s and the
# coverage/Windows suites at ~200s when healthy, so the old 900/300 limits
# left almost no margin on slow shared runners.
hnsw_timeout = is_sanitized ? 1500 : 600

test('Concepts and VectorView', test_concepts)
test('Distance Metrics', test_distances)
test('Utilities (Array)', test_utils)
test('SQLite Functions (Comprehensive)', test_sqlite_functions, timeout: 60)
test('Batch Distance Operations', test_batch_distance)
test('HNSW Index', test_hnsw, timeout: hnsw_timeout)
test('TLS Linkage (Multi-TU)', test_tls_linkage)

test_quantization = executable(
  'test_quantization',
  'test_quantization.cpp',
  include_directories: inc_dirs,
  dependencies: [sqlite3_dep],
  cpp_args: test_cpp_args + ['-DSQLITE_VEC_CPP_TESTING'],
)
test('Quantization (LVQ + RaBitQ + Two-Stage HNSW)', test_quantization, timeout: hnsw_timeout)

test_persistence_fuzz = executable(
  'test_persistence_fuzz',
  'test_persistence_fuzz.cpp',
  include_directories: inc_dirs,
  dependencies: [sqlite3_dep],
  cpp_args: test_cpp_args,
)
test('Persistence Fuzz (Corrupt Blobs)', test_persistence_fuzz, timeout: 120)

test_adversarial_vectors = executable(
  'test_adversarial_vectors',
  'test_adversarial_vectors.cpp',
  include_directories: inc_dirs,
  dependencies: [sqlite3_dep],
  cpp_args: test_cpp_args,
)
test('Adversarial Vectors (NaN/Inf/Zero)', test_adversarial_vectors, timeout: 120)

test_overflow = executable(
  'test_overflow',
  'test_overflow.cpp',
  include_directories: inc_dirs,
  sources: [
    'test_overflow.cpp',
    '../src/sqlite_vec_c_api.cpp',
  ],
  dependencies: [sqlite3_dep],
  cpp_args: test_cpp_args + ['-DSQLITE_VEC_CPP_ENABLED'],
)
test('vec0 Overflow & Validation', test_overflow, timeout: 120)
