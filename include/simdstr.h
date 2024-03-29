
#pragma once

#include <stddef.h>
#include <stdbool.h>

float sum_simd(const float *vec, size_t len);
float sum_simd_fast(const float *vec, size_t len);
bool  memcmpeq_autovec(const char *s1, const char *s2, size_t len);
bool  memcmpeq_avx2(const char *s1, const char *s2, size_t len);
bool  memcmpeq_sse(const char *s1, const char *s2, size_t len);
bool  memcmpeq_sse4_2(const char *s1, const char *s2, size_t len);
bool  memcmpeq_sse4_2_fast(const char *s1, const char *s2, size_t len);
bool  memcmpeq_avx512(const char *s1, const char *s2, size_t len);
char* tolower_simd(char *dst, const char *src, size_t len);
int   compact_simd(char *dst, const char *src, size_t len);
int   qstrlen_simd(const char *src, size_t len);
char* strstr_simd(const char *str, size_t n, const char *substr, size_t sn);