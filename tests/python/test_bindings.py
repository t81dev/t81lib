# tests/python/test_bindings.py — Python-side regression tests for the binding layer.

import array
import sys

try:
    import t81lib
except ImportError as exc:
    raise SystemExit(
        "Failed to import the t81lib Python module. "
        "Build the project with -DT81LIB_BUILD_PYTHON_BINDINGS=ON and "
        "add the build directory to PYTHONPATH."
    ) from exc


def run_gemm_tests() -> bool:
    M, N, K = 2, 3, 48
    K_limbs = K // 48
    limb_pattern = bytes([13] * 16)

    def make_packed_buffer(count: int):
        storage = bytearray(limb_pattern * count)
        return storage, memoryview(storage)

    a_storage, a_view = make_packed_buffer(M * K_limbs)
    b_storage, b_view = make_packed_buffer(K_limbs * N)

    c_values = array.array("f", [0.0] * (M * N))
    t81lib.gemm_ternary(a_view, b_view, c_values, M, N, K)
    if any(abs(value) > 1e-6 for value in c_values):
        return False

    c_values = array.array("f", [2.0] * (M * N))
    t81lib.gemm_ternary(a_view, b_view, c_values, M, N, K, alpha=0.0, beta=0.5)
    if any(abs(value - 1.0) > 1e-6 for value in c_values):
        return False

    return True


def main() -> int:
    small = t81lib.BigInt(123)
    negative = t81lib.BigInt(-45)
    combined = small + negative
    if str(combined) != "78":
        print("Addition sanity check failed", file=sys.stderr)
        return 1
    gcd_value = t81lib.BigInt.gcd(t81lib.BigInt(48), t81lib.BigInt(18))
    if str(gcd_value) != "6":
        print("GCD sanity check failed", file=sys.stderr)
        return 1
    if (t81lib.one() & t81lib.one()) != t81lib.one():
        print("Bitwise AND sanity check failed", file=sys.stderr)
        return 1
    if not run_gemm_tests():
        print("GEMM binding sanity check failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
