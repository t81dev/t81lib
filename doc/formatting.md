# Formatting & Linting

t81lib enforces a consistent C++ style so reviewers can focus on the algorithms.

## clang-format

We ship a repository-wide `.clang-format` configured in the Google/C++ family with 4-space indents, 100-column wrapping, Allman braces, and pointer alignment on the left. Keep your patches tidy by running:

```bash
find include src tests bench -name '*.hpp' -o -name '*.cpp' \
  | xargs clang-format -i
```

`clang-format` respects `.clang-format` at the project root, so format changes automatically when editing files in place (the `.clang-format` file must accompany commits).

## clang-tidy

Use `clang-tidy` to catch lifetime, readability, and performance issues before review. Run it from the build directory so any `compile_commands.json` is honored:

```bash
cmake -S . -B build
clang-tidy -p build include/t81/core/T81Limb.hpp
```

The repository provides a `.clang-tidy` profile that enables analyzer, modernize, readability, and performance checks while restricting the scope to the `t81/` public headers.
