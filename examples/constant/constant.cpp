#include <iostream>
#include <immintrin.h>

void print(__m256i v) {
    int16_t temp[16];
    _mm256_storeu_si256((__m256i*)temp, v);
    for (size_t i = 0; i < 16; i++) {
        std::cout << std::hex << temp[i];
    }
    std::cout << std::endl;
}

int main() {
    __m256i v = _mm256_setzero_si256();

    // set zero
    __m256i zeros = _mm256_setzero_si256();
    print(zeros);

    // set -1
    __m256i minus = _mm256_cmpeq_epi8(v, v);
    print(minus);

    // set 1
    __m256i ones = _mm256_set1_epi16(1);
    print(ones);

    return 0;
}