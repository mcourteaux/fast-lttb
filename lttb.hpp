#pragma once

#include <cstdint>

namespace lttb {

uint64_t downsample(float *x, float *y, uint64_t len, float *out_x,
                    float *out_y, uint64_t out_cap, int bucket_size);

uint64_t downsample(double *x, double *y, uint64_t len, double *out_x,
                    double *out_y, uint64_t out_cap, int bucket_size);

}  // namespace lttb
