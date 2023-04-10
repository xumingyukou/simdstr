#pragma once

#include <immintrin.h>
#include <cstddef>
#include <cstdint>

static int skipspace_naive(const char* json, size_t len) {
    const char* sp = json;
    while (len > 0 && (*sp == ' ' ||  *sp == '\r' || *sp == '\t' || *sp == '\n')) {
        sp++;
        len--;
    }
    return sp - json;
}

static int skipspace_use_cmpeq(const char* json, size_t len) {
    const char* sp = json;
    __m256i spaces = _mm256_set1_epi8(' ');
    __m256i tabs = _mm256_set1_epi8('\t');
    __m256i newlines = _mm256_set1_epi8('\n');
    __m256i carriage_returns = _mm256_set1_epi8('\r');

    while (len >= 32) {
        __m256i input = _mm256_loadu_si256((__m256i*)sp);
        __m256i cmp_spaces = _mm256_cmpeq_epi8(input, spaces);
        __m256i cmp_tabs = _mm256_cmpeq_epi8(input, tabs);
        __m256i cmp_newlines = _mm256_cmpeq_epi8(input, newlines);
        __m256i cmp_carriage_returns = _mm256_cmpeq_epi8(input, carriage_returns);

        __m256i result = _mm256_or_si256(cmp_spaces, cmp_tabs);
        result = _mm256_or_si256(result, cmp_newlines);
        result = _mm256_or_si256(result, cmp_carriage_returns);
        int32_t mask = _mm256_movemask_epi8(result);
        if (mask != -1) {
            break;
        }
        sp += 32;
        len -= 32;
    }
    return sp - json + skipspace_naive(sp, len);
}

// Space: character ' ', ASCII code 0x20
// Tab: character '\t', ASCII code 0x09
// Line Feed/New Line: character '\n', ASCII code 0x0A
// Carriage Return: character '\r', ASCII code 0x0D
static int skipspace_use_shuffle(const char* json, size_t len) {
    const char* sp = json;
    __m256i space_tab = _mm256_setr_epi8(
        '\x20', 0, 0, 0, 0, 0, 0, 0,
         0, '\x09', '\x0A', 0, 0, '\x0D', 0, 0,
        '\x20', 0, 0, 0, 0, 0, 0, 0,
         0, '\x09', '\x0A', 0, 0, '\x0D', 0, 0
    );

    while (len >= 32) {
        __m256i input = _mm256_loadu_si256((__m256i*)sp);
        __m256i shuffle = _mm256_shuffle_epi8(space_tab, input);
        __m256i result = _mm256_cmpeq_epi8(input, shuffle);
        int32_t mask = _mm256_movemask_epi8(result);
        if (mask != -1) {
            break;
        }
        sp  += 32;
        len -= 32;
    }
    return sp - json + skipspace_naive(sp, len);
}

