#pragma once

#include <exception>
#include <functional>
#include <iostream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace Catch {
struct TestFailure : std::runtime_error {
    TestFailure(const std::string& expr, const char* file, int line)
        : std::runtime_error(expr + " (" + file + ":" + std::to_string(line) + ")")
    {}
};

struct TestCase {
    std::string name;
    std::function<void()> fn;
};

inline std::vector<TestCase>& registered_tests() {
    static std::vector<TestCase> tests;
    return tests;
}

inline int run_all(std::ostream& out = std::cout, std::ostream& err = std::cerr) {
    int failed = 0;
    out << "[info] Running " << registered_tests().size() << " test cases\n";
    for (const auto& test : registered_tests()) {
        try {
            test.fn();
            out << "[pass] " << test.name << "\n";
        } catch (const TestFailure& ex) {
            ++failed;
            err << "[fail] " << test.name << ": " << ex.what() << "\n";
        } catch (const std::exception& ex) {
            ++failed;
            err << "[fail] " << test.name << ": unexpected exception: " << ex.what() << "\n";
        } catch (...) {
            ++failed;
            err << "[fail] " << test.name << ": unknown exception\n";
        }
    }
    if (failed == 0) {
        out << "[info] All tests passed\n";
    } else {
        err << "[info] " << failed << " test(s) failed\n";
    }
    return failed;
}
} // namespace Catch

#define CATCH_INTERNAL_TEST_NAME(id) CATCH_INTERNAL_TEST_##id
#define CATCH_INTERNAL_REGISTRAR_NAME(id) CATCH_INTERNAL_REGISTRAR_##id
#define CATCH_INTERNAL_PASTE_IMPL(a, b) a##b
#define CATCH_INTERNAL_PASTE(a, b) CATCH_INTERNAL_PASTE_IMPL(a, b)

#define CATCH_INTERNAL_TEST_CASE_IMPL(name, tags, id)                             \
    static void CATCH_INTERNAL_TEST_NAME(id)();                                   \
    struct CATCH_INTERNAL_REGISTRAR_NAME(id) {                                    \
        CATCH_INTERNAL_REGISTRAR_NAME(id)() {                                     \
            Catch::registered_tests().push_back({name, CATCH_INTERNAL_TEST_NAME(id)}); \
        }                                                                         \
    };                                                                            \
    static CATCH_INTERNAL_REGISTRAR_NAME(id)                                      \
        CATCH_INTERNAL_PASTE(CATCH_INTERNAL_TEST_NAME(id), _registrar);            \
    static void CATCH_INTERNAL_TEST_NAME(id)()

#define TEST_CASE(name, tags) CATCH_INTERNAL_TEST_CASE_IMPL(name, tags, __LINE__)

#define REQUIRE(expr)                                                            \
    do {                                                                         \
        if (!(expr)) throw Catch::TestFailure(#expr, __FILE__, __LINE__);        \
    } while (0)

#define REQUIRE_THROWS_AS(expr, type)                                             \
    do {                                                                         \
        bool caught = false;                                                      \
        try {                                                                     \
            expr;                                                                 \
        } catch (const type&) {                                                   \
            caught = true;                                                        \
        } catch (...) {                                                           \
            throw Catch::TestFailure("caught unexpected exception", __FILE__, __LINE__); \
        }                                                                         \
        if (!caught) throw Catch::TestFailure("expected exception not thrown", __FILE__, __LINE__); \
    } while (0)

#ifdef CATCH_CONFIG_MAIN
int main() {
    return Catch::run_all();
}
#endif
