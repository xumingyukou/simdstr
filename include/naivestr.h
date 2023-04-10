#pragma once

#include <stddef.h>
#include <stdbool.h>

// native functions
float sum_naive(const float *vec, size_t len);
bool  memcmpeq_naive(const char *s1, const char *s2, size_t len);
char* tolower_naive(char *dst, const char *src, size_t len);
int   compact_naive(char *dst, const char *src, size_t len);
int   qstrlen_naive(const char *src, size_t len);
char* strstr_naive(const char *str, size_t n, const char *subtr, size_t sn);