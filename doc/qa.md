# QA checklist

Use this checklist when triaging regressions or when approving contributions to ensure the ternary math stays solid:

1. **Build hygiene**
   - `cmake -S . -B build -DENABLE_SANITIZERS=ON`
   - `cmake --build build`
   - `cmake --build build --target test`
   - If sanitizers report issues, start a focused reproduction with `ctest --rerun-failed --output-on-failure`.

2. **Regression coverage**
   - Run `ctest -R unit_tests` to verify addition/negation/subtraction/karatsuba invariants.
   - Run `ctest -R fuzz` after symbol changes that affect multiplication or carry maps.
   - Confirm new additions have matching canonical data (e.g., `mul_wide_fast` vs `mul_wide_canonical`).

3. **Memory/undefined behavior**
   - When landing new code, enable `ENABLE_SANITIZERS` (see above) to catch buffer overruns, UB, and memory leaks.
   - If you rely on sanitizers, keep the debug build fast by limiting iterations (the fuzz target already runs 1k iterations).

4. **Documentation**
   - Update `doc/api.md` with any new user-visible helpers or recipes.
   - If APIs change, regenerate Doxygen (`doxygen Doxyfile`) and check `docs/html/index.html`.

5. **Release readiness**
   - Increment `project(... VERSION ...)` in `CMakeLists.txt` and add an entry to `CHANGELOG.md` describing the change.
   - Verify GitHub Actions (CI, performance, release) pass on the `main` branch before tagging a release.
