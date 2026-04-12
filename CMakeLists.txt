cmake_minimum_required(VERSION 3.20)

project(sqlite-vec-cpp-bootstrap LANGUAGES NONE)

set(SQLITE_VEC_CPP_MESON_BUILD_DIR "${CMAKE_SOURCE_DIR}/build" CACHE PATH "Meson build directory")
set(SQLITE_VEC_CPP_BUILD_TYPE "debugoptimized" CACHE STRING "Meson buildtype")
set_property(CACHE SQLITE_VEC_CPP_BUILD_TYPE PROPERTY STRINGS debug plain debugoptimized release minsize)
set(SQLITE_VEC_CPP_CPP_STD "c++20" CACHE STRING "C++ standard forwarded to Meson")
set_property(CACHE SQLITE_VEC_CPP_CPP_STD PROPERTY STRINGS c++20 c++23)

option(SQLITE_VEC_CPP_ENABLE_TESTS "Build unit tests" ON)
option(SQLITE_VEC_CPP_ENABLE_BENCHMARKS "Build benchmark suite" OFF)
option(SQLITE_VEC_CPP_ENABLE_SIMD_AVX "Enable x86 AVX-family tuning" OFF)
option(SQLITE_VEC_CPP_ENABLE_SIMD_NEON "Enable baseline AArch64 NEON" ON)
option(SQLITE_VEC_CPP_ENABLE_SIMD_ARM32_NEON "Enable 32-bit ARM NEON tuning" OFF)
option(SQLITE_VEC_CPP_ENABLE_SIMD_DOTPROD "Enable AArch64 DotProd tuning" OFF)

set(SQLITE_VEC_CPP_EXTRA_MESON_ARGS "" CACHE STRING "Additional arguments forwarded to Meson setup")

find_program(MESON_EXECUTABLE meson REQUIRED)

if(SQLITE_VEC_CPP_ENABLE_TESTS)
  set(_sqlite_vec_cpp_enable_tests true)
else()
  set(_sqlite_vec_cpp_enable_tests false)
endif()

if(SQLITE_VEC_CPP_ENABLE_BENCHMARKS)
  set(_sqlite_vec_cpp_enable_benchmarks true)
else()
  set(_sqlite_vec_cpp_enable_benchmarks false)
endif()

if(SQLITE_VEC_CPP_ENABLE_SIMD_AVX)
  set(_sqlite_vec_cpp_enable_simd_avx true)
else()
  set(_sqlite_vec_cpp_enable_simd_avx false)
endif()

if(SQLITE_VEC_CPP_ENABLE_SIMD_NEON)
  set(_sqlite_vec_cpp_enable_simd_neon true)
else()
  set(_sqlite_vec_cpp_enable_simd_neon false)
endif()

if(SQLITE_VEC_CPP_ENABLE_SIMD_ARM32_NEON)
  set(_sqlite_vec_cpp_enable_simd_arm32_neon true)
else()
  set(_sqlite_vec_cpp_enable_simd_arm32_neon false)
endif()

if(SQLITE_VEC_CPP_ENABLE_SIMD_DOTPROD)
  set(_sqlite_vec_cpp_enable_simd_dotprod true)
else()
  set(_sqlite_vec_cpp_enable_simd_dotprod false)
endif()

set(_sqlite_vec_cpp_meson_args
  setup
  "${SQLITE_VEC_CPP_MESON_BUILD_DIR}"
  "${CMAKE_SOURCE_DIR}"
  "--buildtype=${SQLITE_VEC_CPP_BUILD_TYPE}"
  "-Dcpp_std=${SQLITE_VEC_CPP_CPP_STD}"
  "-Denable_tests=${_sqlite_vec_cpp_enable_tests}"
  "-Denable_benchmarks=${_sqlite_vec_cpp_enable_benchmarks}"
  "-Denable_simd_avx=${_sqlite_vec_cpp_enable_simd_avx}"
  "-Denable_simd_neon=${_sqlite_vec_cpp_enable_simd_neon}"
  "-Denable_simd_arm32_neon=${_sqlite_vec_cpp_enable_simd_arm32_neon}"
  "-Denable_simd_dotprod=${_sqlite_vec_cpp_enable_simd_dotprod}"
)

if(SQLITE_VEC_CPP_BUILD_TYPE STREQUAL "debugoptimized")
  list(APPEND _sqlite_vec_cpp_meson_args "-Db_ndebug=true")
endif()

if(EXISTS "${SQLITE_VEC_CPP_MESON_BUILD_DIR}/meson-private")
  list(INSERT _sqlite_vec_cpp_meson_args 1 --reconfigure)
endif()

if(SQLITE_VEC_CPP_EXTRA_MESON_ARGS)
  separate_arguments(_sqlite_vec_cpp_extra_args NATIVE_COMMAND "${SQLITE_VEC_CPP_EXTRA_MESON_ARGS}")
  list(APPEND _sqlite_vec_cpp_meson_args ${_sqlite_vec_cpp_extra_args})
endif()

execute_process(
  COMMAND "${MESON_EXECUTABLE}" ${_sqlite_vec_cpp_meson_args}
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  RESULT_VARIABLE _sqlite_vec_cpp_meson_result
)

if(NOT _sqlite_vec_cpp_meson_result EQUAL 0)
  message(FATAL_ERROR "Meson configure failed with exit code ${_sqlite_vec_cpp_meson_result}")
endif()

add_custom_target(sqlite-vec-cpp ALL
  COMMAND "${MESON_EXECUTABLE}" compile -C "${SQLITE_VEC_CPP_MESON_BUILD_DIR}"
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  USES_TERMINAL
  COMMENT "Building sqlite-vec-cpp via Meson"
)

add_custom_target(sqlite-vec-cpp-install
  COMMAND "${MESON_EXECUTABLE}" install -C "${SQLITE_VEC_CPP_MESON_BUILD_DIR}"
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  DEPENDS sqlite-vec-cpp
  USES_TERMINAL
  COMMENT "Installing sqlite-vec-cpp via Meson"
)
