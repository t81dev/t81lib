# Python installation

This page collects the Python install paths that previously lived in the README.

## Pip wheel / editable installs

```bash
pip install .[torch]
```

This commands runs CMake, builds the pybind11 extension, installs `t81lib` & the higher-level `t81` helper package, and pulls the optional `torch`, `transformers`, and `accelerate` extras when you request `[torch]`. Use a virtualenv when you need to isolate the bindings from system Python.

```bash
pip install .
```

is also supported if you only need the core binding helpers without the PyTorch extras.

## Pipx-managed CLI helpers

Install with `pipx` to keep the CLI tools discovery-separate from your default Python environment:

```bash
pipx install .[torch]
pipx ensurepath
pipx inject t81lib torch transformers accelerate
```

The first command builds the wheel and exposes `t81-convert`, `t81-gguf`, and `t81-qat` in `~/.local/bin`. Use `pipx inject` to add or refresh optional dependencies later.

## Build from source

If you want to inspect the C++ core while keeping the Python bindings in-tree:

```bash
cmake -S . -B build -DT81LIB_BUILD_TESTS=ON -DT81LIB_BUILD_PYTHON_BINDINGS=ON
cmake --build build -j
PYTHONPATH=build python tests/python/test_bindings.py
```

Set `PYTHONPATH=build` to make the just-built extension importable. Run `tests/python/test_bindings.py` as a sanity check; `tests/python/test_gguf.py` exercises the GGUF helpers once `torch`/`transformers` are available.

## Validation tips

- `python -c "import t81lib; print(t81lib.BigInt(42))"` ensures the wheel loads.
- `pipx run t81-convert --help` shows the CLI entry point without installing a second copy.
