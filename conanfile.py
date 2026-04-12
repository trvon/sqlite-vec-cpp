from conan import ConanFile
from conan.tools.gnu import PkgConfigDeps
from conan.tools.meson import MesonToolchain


class SqliteVecCppDepsConan(ConanFile):
    name = "sqlite-vec-cpp-deps"
    version = "0.1"
    settings = "os", "compiler", "build_type", "arch"
    options = {"with_benchmarks": [True, False]}
    default_options = {
        "with_benchmarks": False,
        "benchmark/*:with_libbacktrace": False,
        "benchmark/*:with_libbpf": False,
        "benchmark/*:with_perf_counters": False,
        "benchmark/*:with_pthread": False,
    }

    def requirements(self):
        self.requires("sqlite3/3.46.1")
        if self.options.with_benchmarks:
            self.requires("benchmark/1.9.4")

    def generate(self):
        MesonToolchain(self).generate()
        PkgConfigDeps(self).generate()
