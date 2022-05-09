#pragma once

#include <cstdint>

namespace lttb {

uint64_t downsample(const float *x, const float *y, uint64_t len, float *out_x,
                    float *out_y, uint64_t out_cap, int bucket_size);

uint64_t downsample(const double *x, const double *y, uint64_t len,
                    double *out_x, double *out_y, uint64_t out_cap,
                    int bucket_size);

uint64_t downsample_simd(const float *x, const float *y, uint64_t len,
                         float *out_x, float *out_y, uint64_t out_cap,
                         int bucket_size);

uint64_t downsample_simd(const double *x, const double *y, uint64_t len,
                         double *out_x, double *out_y, uint64_t out_cap,
                         int bucket_size);

}  // namespace lttb
