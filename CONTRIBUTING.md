# Contributing to t81lib

Thanks for your interest. To contribute:

1. Fork the repository and create a feature branch.
2. Run `cmake -S . -B build && cmake --build build` to verify the interface builds.
3. Run `cmake --build build --target test` to ensure the smoke test passes.
4. If you are adding benchmarks, enable `BUILD_BENCHMARKS=ON` and run `./build/bench/t81lib-bench`.
5. Follow the coding style of the existing headers (clang-format compatible) and keep commits focused.

Submit a pull request with a clear description, mention any API changes in `CHANGELOG.md`, and link the relevant docs/benchmarks.
