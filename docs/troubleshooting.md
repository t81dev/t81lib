<!--
docs/troubleshooting.md — Collection of quick fixes and diagnostics for common pain points.
-->

# Troubleshooting

This guide complements [README.md](../README.md) and the other reference pages by pointing you to first steps when configure/build/install/test flows misbehave.

## CMake configure or build failures

- **Check the toolchain.** Run `cmake --version` and `clang++ --version`/`g++ --version` so CMake can emit a clear error about C++20 support. Upgrade to at least CMake 3.22 if you see `CMAKE_CXX_STANDARD` complaints.
- **Clean stale state.** Remove the stale `build/` directory (or `build-full/` if you were using CUDA/ROCm) before rerunning `cmake -S . -B build -DT81LIB_BUILD_TESTS=ON`. Problems that look like “header not found” often come from mixing generator outputs.
- **Missing dependency hints.** When configure mentions `pybind11` or other dependencies, confirm you ran `pip install -U pip setuptools wheel` (needed before `pip install ".[torch]"`) and avoid mixing system and virtualenv interpreters.
- **Enable diagnostics.** Build with `cmake --build build --target t81lib --config RelWithDebInfo -j` and capture the verbose log; redirect into `build/verb.log` if the console output scrolls too fast.

## Python binding & pip install issues

- **Virtual environment isolation.** Activate the same interpreter you used for `cmake` (check `python -m site`) before running `pip install ".[torch]"`. The binding build looks at the interpreter’s include/lib paths.
- **Make sure pybind11 is fresh.** The repo ships `python/bindings.cpp` and uses the `pybind11` submodule; a stale pip package can cause symbol mismatches. Run `pip install -U pybind11` inside the venv if `ImportError: undefined symbol` occurs.
- **pipx helpers.** For CLI helpers installed via `pipx`, re-run `pipx inject t81lib torch transformers accelerate datasets safetensors` after `git pull` so the optional dependencies stay aligned with your working tree.
- **Confirm the binding works.** Run:

  ```bash
  python - <<'PY'
  import t81lib
  print("t81lib", t81lib.__version__)
  PY
  ```

  Any errors here usually mean the `.so`/`.dylib` built against a different Python than the one in `PATH`.

## CLI helpers (`t81`, `t81-qat`, `t81-dequant`)

- **Not found / wrong version.** If `command not found` or the CLI reports a stale version, ensure your `PATH` includes the venv’s `bin/` directory or the pipx shim (`~/.local/bin`). Reinstall via `pip install .[torch]` or `pipx reinstall t81lib`.
- **Validation failures.** The `--validate` flag reruns `gguf.read_gguf`. If it fails, rerun with `--validate --verbose` to surface metadata problems, and verify the GGUF file with `llama.cpp` tooling (`gguf_validate`, `gguf_to_gguf`).
- **Progress bar missing?** The progress reporting relies on `tqdm`; install it (`pip install tqdm`) if the CLI skips bars or prints raw percentages.
- **Meta device / accelerate offload errors.** When converting large Hugging Face checkpoints with the default `device_map=auto`, Accelerate may place many layers onto disk/`meta`. If `t81 convert`/`t81 gguf` (or the legacy `t81-convert`/`t81-gguf` scripts) later tries to call `.to("cpu")` you’ll hit `NotImplementedError: Cannot copy out of meta tensor` or `RuntimeError: You can't move a model that has some modules offloaded to cpu or disk.` Always rerun with `--force-cpu-device-map` or `--device-map none/cpu` so the checkpoints stay on host RAM, and set `ACCELERATE_DISABLE=1` or `HF_ACCELERATE_DISABLE=1` before launching the CLI so no accelerate hooks re-enable offloading. This makes every `nn.Linear` serializable and avoids the meta-device save failure that occurs after the “Some parameters are on the meta device” log.
- **Large GGUF conversions.** Extremely large ternary bundles (Gemma 3.x / Llama 3.x) may exhaust RAM when you read them with older readers because the whole file was loaded before parsing. The new `t81.gguf.read_gguf` implementation parses metadata, tensor infos, and tensor payloads directly from the file handle, seeks to each sorted tensor offset, and never slices the entire bundle into memory. When you still hit memory pressure or Matplotlib font-cache warnings, define `MPLCONFIGDIR=$PWD/data/cache/matplotlib` and `FONTCONFIG_PATH=$PWD/data/cache/fontconfig`, prefer `--force-cpu-device-map`, and keep `ACCELERATE_DISABLE=1`/`HF_ACCELERATE_DISABLE=1` set before rerunning the CLI so every tensor stays on the CPU.

## Testing & benchmarking hiccups

- **ctest fails.** Rebuild with `cmake --build build --target t81-tests` and rerun `ctest --test-dir build --output-on-failure`. Capture `tests/unit/test_output.txt` (if created) as part of the diagnostics.
- **Python tests fail.** From the repo root:

  ```bash
  python -m pytest tests/python/test_bindings.py
  ```

  Pass `-k <pattern>` to narrow it down or `-vv` for full stack. Virtualenv mismatch shows as `ModuleNotFoundError: No module named 't81lib'`.
- **Benchmark hangs.** The Fashion-MNIST benchmark (`scripts/ternary_quantization_benchmark.py`) logs latency/compression; if it stalls, check for GPU dispatch issues by setting `USE_CUDA=OFF`/`USE_ROCM=OFF` and rerun with `T81LIB_DISABLE_NEON=1`.

## GPU & device-specific problems

- **CUDA/ROCm kernels not used.** Pass `-DUSE_CUDA=ON` or `-DUSE_ROCM=ON` at configure time and make sure the `cuda`/`rocm` toolkits are visible in `CUDA_HOME`/`ROCM_PATH`. Use `cmake -S . -B build -DUSE_CUDA=ON -DT81LIB_BUILD_TESTS=ON` to rebuild the Python extension with the dispatch layer.
- **NEON overriden.** If you want to avoid runtime SIMD dispatch (e.g., cross-compiling for unknown hardware), define `T81_DISABLE_NEON=1` in `CXXFLAGS` before configuring.
- **GPUs missing metadata.** The helpers rely on `t81::TensorMetadata`. If you see `unsupported dtype` while operating on NumPy/Torch tensors, confirm you installed the versions documented in `pyproject.toml` (e.g., `pip install ".[torch]"`) so the binding and Torch/Numpy builds stay aligned.

## Miscellaneous

- **Docs/site builds broken.** The doc site uses `mkdocs`. Run `pip install mkdocs mkdocstrings` and verify `mkdocs serve` works from the repo root before pushing updates that reference new pages.
- **New files not picked up.** After adding source/header files, rerun `cmake --build build` (don’t forget `cmake -S . -B build --refresh-cache` if you moved files between directories) and rerun tests.
- **Need to surface diagnostics.** Capture logs via `script -q -c "cmake --build build && ctest --test-dir build --output-on-failure" test-output/ci.log` so you can share the raw output when reporting issues.

## Where to go next

- Review the [docs/index.md](docs/index.md) portal for deep dives on Python, Torch, CLI, and hardware flows.
- If you hit a reproducible failure, open an issue referencing the log from `build/test-output` and the exact `cmake` command you used.
