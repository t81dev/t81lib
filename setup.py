# setup.py — Build helper that activates the CMake-based pybind11 module.
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self) -> None:
        try:
            subprocess.check_output(["cmake", "--version"], stderr=subprocess.DEVNULL)
        except OSError as exc:
            raise RuntimeError("CMake must be installed to build the extensions") from exc
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext: CMakeExtension) -> None:
        extdir = Path(self.get_ext_fullpath(ext.name)).parent.resolve()
        cfg = "Debug" if self.debug else "Release"
        build_temp = Path(self.build_temp) / ext.name
        build_temp.mkdir(parents=True, exist_ok=True)

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            "-DT81LIB_BUILD_TESTS=OFF",
            "-DT81LIB_BUILD_BENCH=OFF",
            "-DT81LIB_BUILD_PYTHON_BINDINGS=ON",
        ]

        torch_prefix = self._torch_cmake_prefix()
        if torch_prefix:
            cmake_args.append("-DT81LIB_ENABLE_TORCH_BINDINGS=ON")
            cmake_args.append(f"-DTORCH_CMAKE_PREFIX_PATH={torch_prefix}")

        build_args = ["--config", cfg, "--target", "t81lib_python"]

        subprocess.check_call(
            ["cmake", "-S", ext.sourcedir, "-B", str(build_temp), *cmake_args]
        )
        subprocess.check_call(["cmake", "--build", str(build_temp), *build_args])

    def _torch_cmake_prefix(self) -> Optional[str]:
        try:
            import torch
        except ImportError:
            return None
        prefix_data = torch.utils.cmake_prefix_path
        if isinstance(prefix_data, (list, tuple)):
            paths = [str(path) for path in prefix_data if path]
        else:
            raw_value = str(prefix_data)
            paths = [part for part in raw_value.split(os.pathsep) if part]
        return ";".join(paths) if paths else None


this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="t81lib",
    version="0.1.0",
    author="The t81lib Contributors",
    author_email="t81lib@example.com",
    description="Balanced ternary arithmetic primitives for AI research",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/t81dev/t81lib",
    packages=find_packages(include=["t81", "t81.*"]),
    python_requires=">=3.8",
    install_requires=["numpy>=1.25", "matplotlib>=3.7"],
    extras_require={
        "torch": [
            "torch>=2.0",
            "transformers>=4.34",
            "safetensors>=0.3",
            "accelerate>=0.20",
        ],
        "dev": ["pytest", "pybind11>=2.12", "cibuildwheel>=2.15"],
    },
    ext_modules=[CMakeExtension("t81lib", sourcedir=str(this_directory / "python"))],
    cmdclass={"build_ext": CMakeBuild},
    entry_points={
        "console_scripts": [
            "t81-convert = t81.convert:main",
            "t81-qat = t81.scripts.t81_qat:main",
            "t81-gguf = t81.scripts.t81_gguf:main",
        ],
    },
    zip_safe=False,
)
