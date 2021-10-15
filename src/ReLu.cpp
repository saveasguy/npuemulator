#include "ReLu.h"

#include <iostream>

#include <immintrin.h>

void npuemulator::ReLu(Vector src, Vector dst)
{
    if (src.length != dst.length) {
        std::cerr << "npuemulator: ReLu: Wrong lengths!" << std::endl;
        exit(1);
    }
    __m256i zerov = _mm256_setzero_si256();
    int i = src.length;
    for (; i >= 64; i -= 64) {
        __m256i v1 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(src.data));
        __m256i v2 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(src.data + 32));
        src.data += 64;
        v1 = _mm256_max_epi8(v1, zerov);
        v2 = _mm256_max_epi8(v2, zerov);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst.data), v1);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst.data + 32), v2);
        dst.data += 64;
    }
    if (i) {
        int8_t data[64];
        memset(data, 0, i);
        memcpy(data, src.data, i);
        __m256i v1 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(data));
        __m256i v2 = _mm256_lddqu_si256(reinterpret_cast<const __m256i *>(data + 32));
        v1 = _mm256_max_epi8(v1, zerov);
        v2 = _mm256_max_epi8(v2, zerov);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(data), v1);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(data + 32), v2);
        memcpy(dst.data, data, i);
    }
}