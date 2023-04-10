#include <stdbool.h>
#include <stddef.h>

#include "naivestr.h"

float sum_naive(const float *arr, size_t len) {
    float sum = 0.0;
    for (size_t i = 0; i < len; i++) {
        sum += arr[i];
    }
    return sum;
}

bool memcmpeq_naive(const char *s1, const char *s2, size_t len) {
    while (len > 0 && *s1++ == *s2++) len--;
    return len == 0;
}

// Convert src to lower case and copy to dst, src is a ASCII string.
char* tolower_naive(char *dst, const char *src, size_t len) {
    for (size_t i = 0; i < len; i++) {
        bool is_upper = (src[i] >= 'A' && src[i] <= 'Z');
        dst[i] =  is_upper ? src[i] + 32 : src[i];
    }
    return dst;
}

// Remove whitespaces from src and copy to dst. Whitespace is defined as ' ', '\t', '\r', '\n'.
// return the length of dst.
int compact_naive(char *dst, const char *src, size_t len) {
    int j = 0;
    for (size_t i = 0; i < len; i++) {
        bool is_space = (src[i] == ' ' || src[i] == '\t' 
                        || src[i] == '\r' || src[i] == '\n');
        if (!is_space) {
            dst[j++] = src[i];
        }
    }
    return j;
}

// Get the unquoted length from the quoted string.
// Return the length of dst if success and -1 if failed.
// In quoted string, '\' or '"' MUST be escaped as "\\\\" or "\\\"".
// Quote string is: 
// "\"abc\""       -> unquoted is "abc", unquoted len is 3
// "\"abcd\\\"\""  -> unquoted is "abc\"", unquoted len is 4
// "\"abcd\\\"\"xxx"  -> unquoted is "abc\"", unquoted len is 4, ignore the trailing
// "\"abc\\a\""    -> invalid unquoted string, '\\' must be escaped
int qstrlen_naive(const char *src, size_t len) {
    int count = 0;

    // check the start quote
    if (len == 0 || src[0] != '"') {
        return -1;
    }

    for (size_t i = 1; i < len; i++) {
        // deal with the escaped chars
        if (src[i] == '\\') {
            // EOF error
            if (i + 1 >= len) {
                return -1;
            }
            // Invalid escaped chars
            if (src[i+1] != '\\' && src[i+1] != '"') {
                return -1;
            }
            i++;
            count++;
            continue;
        }
        // found the ending quotes
        if (src[i] == '"') {
            return count;
        }
        count++;
    }

    // not found the ending quotes
    return -1;
}

// Match the substr in str, return the pointer of found substr in str.
// If substr is empty, return the pointer of str.
char* strstr_naive(const char *str, size_t n, const char *substr, size_t sn) {
    if (sn == 0 || sn > n) {
        return (char*)str;
    }
    for (size_t i = 0; i < n; i++) {
        if (str[i] == substr[0]) {
            bool is_match = true;
            for (size_t j = 1; j < sn; j++) {
                if (str[i + j] != substr[j]) {
                    is_match = false;
                    break;
                }
            }
            if (is_match) {
                return (char*)str + i;
            }
        }
    }
    return NULL;
}
