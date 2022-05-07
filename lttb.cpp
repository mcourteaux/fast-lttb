// vim: sw=2 ts=2 sts=2 expandtab:
#include "lttb.hpp"

#include <stdexcept>

namespace lttb {

template <typename T>
static inline uint64_t downsample0(T *in_x, T *in_y, uint64_t len, T *out_x,
                                   T *out_y, uint64_t out_cap,
                                   int bucket_size) {
  if ((bucket_size % 8) != 0 || bucket_size <= 0) {
    throw std::invalid_argument("bucket_size not a positive multiple of 8");
  }
  uint64_t num_full_buckets = (len - 2) / bucket_size;
  std::printf("len: %lu\n", len);
  std::printf("bucket_size: %d\n", bucket_size);
  std::printf("out_cap: %lu\n", out_cap);
  std::printf("buckets: %lu\n", num_full_buckets);
  if (num_full_buckets + 2 > out_cap) {
    throw std::invalid_argument("out_cap not big enough");
  }

  T last_x = 0;
  if (in_x) {
    last_x = in_x[0];
    in_x++;
  }
  if (out_x) {
    out_x[0] = last_x;
  }
  out_x++;

  T last_y = in_y[0];
  out_y[0] = last_y;
  out_y++;
  in_y++;

  for (uint64_t bi = 0; bi < num_full_buckets; ++bi) {
    short best_si = 0;
    T largest_surface = 0;
    T best_x = 0;
    T best_y = 0;

    T next_x = 0;
    T next_y = 0;
    if (bi + 1 < num_full_buckets) {
      // Compute average of next bucket.
      for (short si = 0; si < bucket_size; ++si) {
        next_y += in_y[bucket_size + si];
        if (in_x) {
          next_x += in_x[bucket_size + si];
        }
      }
      if (in_x) {
        next_x /= bucket_size;
      } else {
        next_x = bucket_size + (bucket_size >> 1);
      }
      next_y /= bucket_size;
    } else {
      // There is no next full bucket. Let's take the last one
      int last_elem_idx = len - num_full_buckets * bucket_size -
                          2;  // -1 extra for the first elemented shifted away
      if (in_x) {
        next_x = in_x[last_elem_idx];
      } else {
        next_x = last_elem_idx;
      }
      next_y = in_y[last_elem_idx];
    }

    for (short si = 0; si < bucket_size; ++si) {
      // Take x,y coordinate of candidate
      T cand_y = in_y[si];
      T cand_x = 0;
      if (in_x) {
        cand_x = in_x[si];
      } else {
        cand_x = bi * bucket_size + si;
      }

      // Calculating surface of triangle using determinant method
      //         [a_x, a_y, 1]
      // abs(det([b_x, b_y, 1])
      //         [c_x, c_y, 1]
      // But first, let's simplify by moving the triangle to have a_x=a_y=0:
      //         [      0,       0, 1]
      // abs(det([b_x-a_x, b_y-a_y, 1]))
      //         [c_x-a_x, c_y-a_y, 1]
      // Which mathematically is the same, but has higher precision.

      T d1_x = cand_x - last_x;
      T d2_x = next_x - last_x;
      T d1_y = cand_y - last_y;
      T d2_y = next_y - last_y;
      T surf = std::abs(d1_x * d2_y - d1_y * d1_x);

      if (surf > largest_surface) {
        largest_surface = surf;
        best_si = si;
        best_x = cand_x;
        best_y = cand_y;
      }
    }

    // Produce an output
    if (out_x) out_x[bi] = best_x;
    out_y[bi] = best_y;

    // Advance to next bucket of data. We will move as much values
    // backwards to make variables hold small numbers.
    last_y = best_y;
    last_x = best_x;
    if (in_x) {
      in_x += bucket_size;
    } else {
      last_x -= bucket_size;
    }
    in_y += bucket_size;
  }

  // Just store the last element.
  int last_elem_idx = len - num_full_buckets * bucket_size - 2;
  if (out_x) {
    if (in_x) {
      last_x = in_x[last_elem_idx];
    } else {
      last_x = len - 1;
    }
    out_x[num_full_buckets] = last_x;
  }
  out_y[num_full_buckets] = in_y[last_elem_idx];

  return num_full_buckets + 2;
}

uint64_t downsample(float *x, float *y, uint64_t len, float *out_x,
                    float *out_y, uint64_t out_cap, int bucket_size = -1) {
  return downsample0<float>(x, y, len, out_x, out_y, out_cap, bucket_size);
}
uint64_t downsample(double *x, double *y, uint64_t len, double *out_x,
                    double *out_y, uint64_t out_cap, int bucket_size = -1) {
  return downsample0<double>(x, y, len, out_x, out_y, out_cap, bucket_size);
}
}  // namespace lttb
