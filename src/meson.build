# SQLite-vec C++ library sources (mostly header-only, but may have implementation files)

sqlite_vec_cpp_sources = []

# C API compatibility layer (if needed for implementation)
if fs.is_file('sqlite_vec_c_api.cpp')
  sqlite_vec_cpp_sources += ['sqlite_vec_c_api.cpp']
endif

# Build library
# Even though mostly header-only, declare a library for proper dependency management
sqlite_vec_cpp_lib = static_library(
  'sqlite-vec-cpp',
  sqlite_vec_cpp_sources.length() > 0 ? sqlite_vec_cpp_sources : [],
  include_directories: inc_dirs,
  dependencies: [sqlite3_dep],
  install: false,  # Internal use in YAMS
  build_by_default: sqlite_vec_cpp_sources.length() > 0
)

# Export dependency (header-only style)
sqlite_vec_cpp_dep = declare_dependency(
  include_directories: inc_dirs,
  link_with: sqlite_vec_cpp_sources.length() > 0 ? sqlite_vec_cpp_lib : [],
  dependencies: [sqlite3_dep],
  compile_args: ['-DSQLITE_VEC_CPP_ENABLED'],
)

# Make available to parent projects
meson.override_dependency('sqlite-vec-cpp', sqlite_vec_cpp_dep)
