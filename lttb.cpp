// vim: sw=2 ts=2 sts=2 expandtab:
#include "lttb.hpp"

#include <immintrin.h>
#include <Tracy.hpp>

#include <stdexcept>

namespace lttb {

template <typename T>
static uint64_t downsample0(const T *in_x, const T *in_y, uint64_t len,
                                   T *out_x, T *out_y, uint64_t out_cap,
                                   int bucket_size) {
  if ((bucket_size % 8) != 0 || bucket_size <= 0) {
    throw std::invalid_argument("bucket_size not a positive multiple of 8");
  }
  uint64_t num_full_buckets = (len - 2) / bucket_size;
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
    out_x++;
  }

  T last_y = in_y[0];
  out_y[0] = last_y;
  out_y++;
  in_y++;

  for (uint64_t bi = 0; bi < num_full_buckets; ++bi) {
    T next_x = 0;
    T next_y = 0;
    if (bi + 1 < num_full_buckets) {
      // Compute average of next bucket.
      for (uint32_t si = 0; si < bucket_size; ++si) {
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

    T largest_surface = 0;
    T best_x = 0;
    T best_y = 0;
    for (uint32_t si = 0; si < bucket_size; ++si) {
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

      if (surf >= largest_surface) {
        largest_surface = surf;
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

template <typename T>
struct simd {
  typedef void type;
  static constexpr int size = 0;
};

#if 1
template <>
struct simd<float> {
  typedef __m256 type;
  typedef int32_t signed_int;
  static constexpr int size = 8;
  static inline type zero() { return _mm256_setzero_ps(); }
  static inline type splat(float v) { return _mm256_set1_ps(v); }
  static inline type load(const float *d) { return _mm256_loadu_ps(d); }
  static inline type add(type a, type b) { return _mm256_add_ps(a, b); }
  static inline type mul(type a, type b) { return _mm256_mul_ps(a, b); }
  static inline type abs(type a) {
    return _mm256_max_ps(a, _mm256_sub_ps(zero(), a));
  }
  static inline type blend(type a, type b, type mask) {
    return _mm256_blendv_ps(a, b, mask);
  }

  static inline type hadd_vec(type a) {
    auto x = _mm256_permute2f128_ps(a, a, 1);
    auto y = _mm256_add_ps(a, x);
    x = _mm256_shuffle_ps(y, y, _MM_SHUFFLE(2, 3, 0, 1));
    x = _mm256_add_ps(x, y);
    y = _mm256_shuffle_ps(x, x, _MM_SHUFFLE(1, 0, 3, 2));
    return _mm256_add_ps(x, y);
  }

  template <int _cmp>
  static inline type cmp(type a, type b) {
    return _mm256_cmp_ps(a, b, _cmp);
  }
};
#else
template <>
struct simd<float> {
  typedef __m128 type;
  typedef int32_t signed_int;
  static constexpr int size = 4;
  static inline type zero() { return _mm_setzero_ps(); }
  static inline type splat(float v) { return _mm_set1_ps(v); }
  static inline type load(const float *d) { return _mm_loadu_ps(d); }
  static inline type add(type a, type b) { return _mm_add_ps(a, b); }
  static inline type mul(type a, type b) { return _mm_mul_ps(a, b); }
  static inline type abs(type a) {
    return _mm_max_ps(a, _mm_sub_ps(zero(), a));
  }
  static inline type blend(type a, type b, type mask) {
    return _mm_blendv_ps(a, b, mask);
  }

  static inline type hadd_vec(type a) {
    type shuf = _mm_movehdup_ps(a);
    type sums = _mm_add_ps(a, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_broadcastss_ps(sums);
  }

  template <int _cmp>
  static inline type cmp(type a, type b) {
    return _mm_cmp_ps(a, b, _cmp);
  }
};
#endif

template <>
struct simd<double> {
  typedef __m256d type;
  typedef int64_t signed_int;
  static constexpr int size = 4;
  static inline type zero() { return _mm256_setzero_pd(); }
  static inline type splat(double v) { return _mm256_set1_pd(v); }
  static inline type load(const double *d) { return _mm256_loadu_pd(d); }
  static inline type add(type a, type b) { return _mm256_add_pd(a, b); }
  static inline type mul(type a, type b) { return _mm256_mul_pd(a, b); }
  static inline type abs(type a) {
    return _mm256_max_pd(a, _mm256_sub_pd(zero(), a));
  }
  static inline type blend(type a, type b, type mask) {
    return _mm256_blendv_pd(a, b, mask);
  }
  static inline type hadd_vec(type a) {
    a = _mm256_hadd_pd(a, _mm256_permute2f128_pd(a, a, 1));
    // a = [ a0 + a1, a2 + a3, a2 + a3, a0 + a1 ]
    a = _mm256_hadd_pd(a, a);
    return a;
  }

  template <int _cmp>
  static inline type cmp(type a, type b) {
    return _mm256_cmp_pd(a, b, _cmp);
  }
};

template <typename T>
static uint64_t downsample0_simd(const T *in_x, const T *in_y,
                                        uint64_t len, T *out_x, T *out_y,
                                        uint64_t out_cap, int bucket_size) {
  if ((bucket_size % 8) != 0 || bucket_size <= 0) {
    throw std::invalid_argument("bucket_size not a positive multiple of 8");
  }
  uint64_t num_full_buckets = (len - 2) / bucket_size;
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

  typedef typename simd<T>::type vec_t;

  float ramp_f32_data[simd<T>::size];
  double ramp_f64_data[simd<T>::size];
  int32_t ramp_i32_data[simd<T>::size];
  int64_t ramp_i64_data[simd<T>::size];
  for (int i = 0; i < simd<T>::size; ++i) {
    ramp_f32_data[i] = i;
    ramp_f64_data[i] = i;
    ramp_i32_data[i] = i;
    ramp_i64_data[i] = i;
  }

  vec_t inv_bucket_size = simd<T>::splat(1.0 / bucket_size);

  for (uint64_t bi = 0; bi < num_full_buckets; ++bi) {
    vec_t next_x = simd<T>::zero();
    vec_t next_y = simd<T>::zero();
    if (bi + 1 < num_full_buckets) {
      // Compute average of next bucket.
      for (uint32_t si = 0; si < bucket_size; si += simd<T>::size) {
        next_y = simd<T>::add(next_y, simd<T>::load(&in_y[bucket_size + si]));
        if (in_x) {
          next_x = simd<T>::add(next_x, simd<T>::load(&in_x[bucket_size + si]));
        }
      }
      if (in_x) {
        next_x = simd<T>::hadd_vec(next_x);
        next_x = simd<T>::mul(next_x, inv_bucket_size);
      } else {
        next_x = simd<T>::splat(bucket_size + (bucket_size >> 1));
      }
      next_y = simd<T>::hadd_vec(next_y);
      next_y = simd<T>::mul(next_y, inv_bucket_size);
    } else {
      // There is no next full bucket. Let's take the last one
      int last_elem_idx = len - num_full_buckets * bucket_size -
                          2;  // -1 extra for the first elemented shifted away
      if (in_x) {
        next_x = simd<T>::splat(in_x[last_elem_idx]);
      } else {
        next_x = simd<T>::splat(last_elem_idx);
      }
      next_y = simd<T>::splat(in_y[last_elem_idx]);
    }

    vec_t v_largest_surface = simd<T>::zero();
    vec_t v_best_x = simd<T>::zero();
    vec_t v_best_y = simd<T>::zero();
    for (uint32_t si = 0; si < bucket_size; si += simd<T>::size) {
      // Take x,y coordinate of candidate
      vec_t cand_y = simd<T>::load(&in_y[si]);
      vec_t cand_x = simd<T>::zero();
      if (in_x) {
        cand_x = simd<T>::load(&in_x[si]);
      } else {
        if constexpr (sizeof(T) == 4) {
          cand_x = simd<T>::splat(bi * bucket_size + si) +
                   simd<T>::load(ramp_f32_data);
        } else if constexpr (sizeof(T) == 8) {
          cand_x = simd<T>::splat(bi * bucket_size + si) +
                   simd<T>::load(ramp_f64_data);
        }
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

      vec_t d1_x = cand_x - last_x;
      vec_t d2_x = next_x - last_x;
      vec_t d1_y = cand_y - last_y;
      vec_t d2_y = next_y - last_y;
      vec_t surf = simd<T>::abs(d1_x * d2_y - d1_y * d1_x);

      vec_t comp = simd<T>::template cmp<_CMP_GE_OQ>(surf, v_largest_surface);
      v_largest_surface = simd<T>::blend(v_largest_surface, surf, comp);
      v_best_x = simd<T>::blend(v_best_x, cand_x, comp);
      v_best_y = simd<T>::blend(v_best_y, cand_y, comp);
    }

    // Collapse vector into one final result
    T best_x, best_y;
    T surfaces_array[simd<T>::size];
    T best_x_array[simd<T>::size];
    T best_y_array[simd<T>::size];
    T largest_surface = -1;
    typename simd<T>::signed_int indices_array[simd<T>::size];
    if constexpr (std::is_same_v<T, float> && simd<T>::size == 8) {
      _mm256_storeu_ps(surfaces_array, v_largest_surface);
      _mm256_storeu_ps(best_x_array, v_best_x);
      _mm256_storeu_ps(best_y_array, v_best_y);
    } if constexpr (std::is_same_v<T, float> && simd<T>::size == 4) {
      _mm_storeu_ps(surfaces_array, v_largest_surface);
      _mm_storeu_ps(best_x_array, v_best_x);
      _mm_storeu_ps(best_y_array, v_best_y);
    } else if constexpr (sizeof(T) == sizeof(double)) {
      _mm256_storeu_pd(surfaces_array, v_largest_surface);
      _mm256_storeu_pd(best_x_array, v_best_x);
      _mm256_storeu_pd(best_y_array, v_best_y);
    }
    for (int i = 0; i < simd<T>::size; ++i) {
      T s = surfaces_array[i];
      if (s >= largest_surface) {
        largest_surface = s;
        best_x = best_x_array[i];
        best_y = best_y_array[i];
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

uint64_t downsample(const float *x, const float *y, uint64_t len, float *out_x,
                    float *out_y, uint64_t out_cap, int bucket_size = -1) {
  //ZoneScoped;
  return downsample0<float>(x, y, len, out_x, out_y, out_cap, bucket_size);
}
uint64_t downsample(const double *x, const double *y, uint64_t len,
                    double *out_x, double *out_y, uint64_t out_cap,
                    int bucket_size = -1) {
  //ZoneScoped;
  return downsample0<double>(x, y, len, out_x, out_y, out_cap, bucket_size);
}

uint64_t downsample_simd(const float *x, const float *y, uint64_t len,
                         float *out_x, float *out_y, uint64_t out_cap,
                         int bucket_size = -1) {
  //ZoneScoped;
  return downsample0_simd<float>(x, y, len, out_x, out_y, out_cap, bucket_size);
}
uint64_t downsample_simd(const double *x, const double *y, uint64_t len,
                         double *out_x, double *out_y, uint64_t out_cap,
                         int bucket_size = -1) {
  //ZoneScoped;
  return downsample0_simd<double>(x, y, len, out_x, out_y, out_cap,
                                  bucket_size);
}
}  // namespace lttb
