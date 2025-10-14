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
sqlite3_dep = dependency('sqlite3', version: '>=3.38.0', required: true)

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
