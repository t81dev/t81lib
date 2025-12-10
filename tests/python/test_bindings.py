import sys

try:
    import t81lib
except ImportError as exc:
    raise SystemExit(
        "Failed to import the t81lib Python module. "
        "Build the project with -DT81LIB_BUILD_PYTHON_BINDINGS=ON and "
        "add the build directory to PYTHONPATH."
    ) from exc


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
    return 0


if __name__ == "__main__":
    sys.exit(main())
