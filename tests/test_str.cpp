#include <cctype>
#include <cstring>
#include <random>
#include <string>
#include <gtest/gtest.h>

extern "C" {
    #include  "naivestr.h"
    #include  "simdstr.h"
}

using memcmpeq_t = bool  (*)(const char *s1, const char *s2, size_t len);
using tolower_t  = char* (*)(char *dst, const char *src, size_t len);
using compact_t  = int   (*)(char *dst, const char *src, size_t len);
using qstrlen_t  = int   (*)(const char *src, size_t len);
using strstr_t   = char* (*)(const char *str, size_t n, const char *substr, size_t sn);

static std::string repeat(const std::string s, int n) {
    std::ostringstream os;
    for(int i = 0; i < n; i++) os << s;
    return os.str();
}

void test_memcmpeq(memcmpeq_t memcmpeq) {
    struct MemcmpEqCase {
        std::string s1;
        std::string s2;
        bool expected;
    };
    std::vector<MemcmpEqCase> tests = {
        MemcmpEqCase{"", "", true},
        MemcmpEqCase{std::string("\0", 1), " ", false},
        MemcmpEqCase{" ", " ", true},
        MemcmpEqCase{"hello", "world", false},
        MemcmpEqCase{"hello", "hello", true},
        MemcmpEqCase{std::string(1024, 'x'), std::string(1024, 'x'), true},
        MemcmpEqCase{std::string(1024, 'x'), std::string(1023, 'x') + 'y', false},
    };

    for (const auto& test : tests) {
        std::stringstream log;
        size_t len = std::min(test.s1.size(), test.s2.size());
        bool got = memcmpeq(test.s1.data(), test.s2.data(), len);
        EXPECT_EQ(got, test.expected) << test.s1 << "_" << test.s2;
    }
}

void test_tolower(tolower_t tolower) {
    struct TolowerCase {
        std::string src;
        std::string expected;
    };
    std::vector<TolowerCase> tests = {
        TolowerCase{"", ""},
        TolowerCase{"Hello, World!", "hello, world!"},
        TolowerCase{"12345", "12345"},
        TolowerCase{"ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz"},
        TolowerCase{"中文😁\u0432", "中文😁\u0432"},
        TolowerCase{repeat("ABCDEFGHIJKLMNOPQRSTUVWXYZ", 32),
                    repeat("abcdefghijklmnopqrstuvwxyz", 32)},
        TolowerCase{std::string(1024, 'X') + "中文😁\u0432", std::string(1024, 'x') + "中文😁\u0432"},
    };

    for (const auto& test : tests) {
        const char* src = test.src.data();
        size_t len = test.src.size();
        char  *dst = new char[len + 1];
        char  *got = tolower(dst, src, len);
        got[len] = '\0';
        EXPECT_EQ(got, test.expected) << test.src;
        delete[] dst;
    }
}

void test_compact(compact_t compact) {
    struct CompactCase {
        std::string src;
        std::string expected;
        int expected_len;
    };
    std::vector<CompactCase> tests = {
        CompactCase{"", "", 0},
        CompactCase{" \t\n\r", "", 0},
        CompactCase{"a b", "ab", 2},
        CompactCase{"a\nb", "ab", 2},
        CompactCase{"a\rb", "ab", 2},
        CompactCase{"a\tb", "ab", 2},
        CompactCase{"a b c d e f g h i j k l m n o p q r s t u v w x y z", 
                    "abcdefghijklmnopqrstuvwxyz", 26},
        CompactCase{std::string(1024, ' '), "", 0},
        CompactCase{std::string(1024, ' ') + "a\rb", "ab", 2},
    };

    for (const auto& test : tests) {
        const char* src = test.src.data();
        size_t len = test.src.size();
        char  *dst = new char[len + 1];
        std::memset(dst, '\0', len + 1);
        int got_len = compact(dst, src, len);
        EXPECT_EQ(dst, test.expected) << test.src;
        EXPECT_EQ(got_len, test.expected_len) << test.src;
        delete[] dst;
    }
}

void test_qstrlen(qstrlen_t qstrlen) {
    struct QstrlenCase {
        std::string src;
        int expected;
    };
    std::vector<QstrlenCase> tests = {
        QstrlenCase{"", -1},
        QstrlenCase{"a", -1},
        QstrlenCase{R"(")", -1},
        QstrlenCase{R"("")", 0},

        QstrlenCase{R"("\)", -1},
        QstrlenCase{R"("\")", -1},
        QstrlenCase{R"("\\)", -1},

        QstrlenCase{R"("\\")", 1},
        QstrlenCase{R"("\"")", 1},
        QstrlenCase{R"("\x")", -1},
        
        QstrlenCase{R"("\\\"")", 2},
        QstrlenCase{R"("\\\""xxx)", 2},
        QstrlenCase{R"("\\\""\")", 2},

        QstrlenCase{R"("\"abc\"")", 5},
        QstrlenCase{R"("\"abcd\\\"\"")", 8},
        QstrlenCase{R"("\"abc\\a\")", -1},
        QstrlenCase{R"("\"abc\\\\")", 6},
        QstrlenCase{R"("\"abc\\\\\")", -1},

        QstrlenCase{std::string("\"") + std::string(63, '\\') + R"("abc"x)", 35},
    };
    for (const auto& test : tests) {
        int result = qstrlen(test.src.data(), test.src.size());
        EXPECT_EQ(result, test.expected) << test.src;
    }
}

void test_strstr(strstr_t strstr) {
    struct StrstrCase {
        std::string str;
        std::string substr;
        int subpos;
    };
    std::vector<StrstrCase> tests = {
        StrstrCase{"", "", 0},
        StrstrCase{"hello", "", 0},
        StrstrCase{"hello", "h", 0},
        StrstrCase{"hello", "l", 2},
        StrstrCase{"hello", "o", 4},
        StrstrCase{"hello", "hello", 0},
        StrstrCase{"hello", "world", -1},
        StrstrCase{std::string(1024, 'X'), "XX", 0},
        StrstrCase{std::string(1024, 'X'), "XXY", -1},
    };
    for (const auto& test : tests) {
        char* result = strstr(test.str.data(), test.str.size(), 
                          test.substr.data(), test.substr.size());
        const char* expect = test.subpos >=0 ? test.str.data() + test.subpos : nullptr;
        EXPECT_EQ(result, expect) << test.str << "_" << test.subpos;
    }
}

#define ADD_TEST(func, arch) \
    TEST(func##_##arch, Basic) {    \
        test_##func(func##_##arch); \
    }

ADD_TEST(memcmpeq, naive);
ADD_TEST(memcmpeq, sse);
ADD_TEST(memcmpeq, avx2);
#if __AVX512F__ &&  __AVX512BW__
ADD_TEST(memcmpeq, avx512);
#endif
ADD_TEST(memcmpeq, autovec);

ADD_TEST(tolower, naive);
ADD_TEST(compact, naive);
ADD_TEST(qstrlen, naive);
ADD_TEST(strstr, naive);

// TODO: add test for simd functions
// ADD_TEST(tolower, simd);

#undef ADD_TEST