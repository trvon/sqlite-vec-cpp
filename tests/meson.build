# Tests

test_concepts = executable(
  'test_concepts',
  'test_concepts.cpp',
  include_directories: inc_dirs,
  dependencies: [sqlite3_dep],
)

test_distances = executable(
  'test_distances',
  'test_distances.cpp',
  include_directories: inc_dirs,
  dependencies: [sqlite3_dep],
)

test_utils = executable(
  'test_utils',
  'test_utils.cpp',
  include_directories: inc_dirs,
  dependencies: [sqlite3_dep],
)

test_sqlite_functions = executable(
  'test_sqlite_functions',
  'test_sqlite_functions.cpp',
  include_directories: inc_dirs,
  sources: [
    'test_sqlite_functions.cpp',
    '../src/sqlite_vec_c_api.cpp',
  ],
  dependencies: [sqlite3_dep, dependency('sqlite-vec-cpp')],
  cpp_args: ['-DSQLITE_VEC_CPP_TESTING'],
)

test_batch_distance = executable(
  'test_batch_distance',
  'test_batch_distance.cpp',
  include_directories: inc_dirs,
  dependencies: [sqlite3_dep],
)

test_hnsw = executable(
  'test_hnsw',
  'test_hnsw.cpp',
  include_directories: inc_dirs,
  dependencies: [sqlite3_dep],
)

test('Concepts and VectorView', test_concepts)
test('Distance Metrics', test_distances)
test('Utilities (Array)', test_utils)
test('SQLite Functions (Comprehensive)', test_sqlite_functions, timeout: 60)
test('Batch Distance Operations', test_batch_distance)
test('HNSW Index', test_hnsw, timeout: 300)
