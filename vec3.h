/*  Supplemental  3-vector  code  that  abstracts  away  a  bunch  of   SIMD
 *  operations.  I  don't  plan  to use this code elsewhere so I chose short
 *  simple names without worrying about name collisions.
 */
#ifndef JELLY_MCJELLYFACE_VEC3
#define JELLY_MCJELLYFACE_VEC3

#ifndef __SSE__
#include "vec3-fallback.inc"
#else
#include "xmmintrin.h"

typedef struct vec3 { __m128 m; } Vec3;

static inline __m128 all_x(__m128 m) { return _mm_shuffle_ps(m, m, 0x00); }
static inline __m128 all_y(__m128 m) { return _mm_shuffle_ps(m, m, 0x55); }
static inline __m128 all_z(__m128 m) { return _mm_shuffle_ps(m, m, 0xAA); }

static inline float X(Vec3 a) { return _mm_cvtss_f32(all_x(a.m)); }
static inline float Y(Vec3 a) { return _mm_cvtss_f32(all_y(a.m)); }
static inline float Z(Vec3 a) { return _mm_cvtss_f32(all_z(a.m)); }

static inline Vec3 vec3(float x, float y, float z) {
    return (Vec3) { _mm_setr_ps(x, y, z, 1.0f) };
}

static inline Vec3 add(Vec3 a, Vec3 b) {
    return (Vec3) { _mm_add_ps(a.m, b.m) };
}

static inline void iadd(Vec3* a, Vec3 b) {
    *a = add(*a, b);
}

static inline Vec3 sub(Vec3 a, Vec3 b) {
    return (Vec3) { _mm_sub_ps(a.m, b.m) };
}

static inline void isub(Vec3* a, Vec3 b) {
    *a = sub(*a, b);
}

static inline Vec3 scale(float c, Vec3 a) {
    return (Vec3) { _mm_mul_ps(_mm_set1_ps(c), a.m) };
}

static inline void iscale(Vec3* a, float c) {
    *a = scale(c, *a);
}

static inline Vec3 elementwise_mul(Vec3 a, Vec3 b) {
    return (Vec3) { _mm_mul_ps(a.m, b.m) };
}

static inline float dot(Vec3 a, Vec3 b) {
    Vec3 product = elementwise_mul(a, b);
    return X(product) + Y(product) + Z(product);
}

static inline Vec3 cross(Vec3 a, Vec3 b) {
    __m128 a_yzx = _mm_shuffle_ps(a.m, a.m, 0x09);
    __m128 a_zxy = _mm_shuffle_ps(a.m, a.m, 0x12);
    __m128 b_zxy = _mm_shuffle_ps(b.m, b.m, 0x12);
    __m128 b_yzx = _mm_shuffle_ps(b.m, b.m, 0x09);
    
    __m128 m = _mm_sub_ps(_mm_mul_ps(a_yzx, b_zxy), _mm_mul_ps(a_zxy, b_yzx));
    return (Vec3) { m };
}

static inline __m128 all_sum_squares(Vec3 a) {
    __m128 squares = _mm_mul_ps(a.m, a.m);
    __m128 sum_xy = _mm_add_ps(all_x(squares), all_y(squares));
    return _mm_add_ps(sum_xy, all_z(squares));
}

static inline float sum_squares(Vec3 a) {
    return _mm_cvtss_f32(all_sum_squares(a));
}

// The next few functions use rsqrt so they aren't all that accurate but
// they are FAAAAST when compiled to native code (rsqrt = 4 clock cycles = 1ns)
// The mask exists to zero-out the answer in case of zero vectors.
static inline float magnitude(Vec3 a) {
    __m128 SS = all_sum_squares(a);
    __m128 mask = _mm_cmpgt_ps(SS, _mm_setzero_ps());
    return _mm_cvtss_f32(_mm_and_ps(mask, _mm_mul_ss(SS, _mm_rsqrt_ss(SS))));
}

static inline Vec3 normalize(Vec3 a) {
    __m128 SS = all_sum_squares(a);
    __m128 mask = _mm_cmpgt_ps(SS, _mm_setzero_ps());
    return (Vec3) { _mm_and_ps(mask, _mm_mul_ps(_mm_rsqrt_ps(SS), a.m)) };
}

static inline Vec3 normalize_magnitude(Vec3 a, float* magnitude) {
    __m128 SS = all_sum_squares(a);
    __m128 mask = _mm_cmpgt_ps(SS, _mm_setzero_ps());
    __m128 rsqrt = _mm_rsqrt_ps(SS);
    *magnitude = _mm_cvtss_f32(_mm_and_ps(mask, _mm_mul_ss(SS, rsqrt)));
    return (Vec3) { _mm_and_ps(mask, _mm_mul_ps(rsqrt, a.m)) };
}

// Return control >= 0 ? x : 0
static inline float step(float x, float control) {
    __m128 mask = _mm_cmpge_ss(_mm_set_ss(control), _mm_setzero_ps());
    return _mm_cvtss_f32(_mm_and_ps(mask, _mm_set_ss(x)));
}

// Return control >= 0 ? positive : negative
static inline Vec3 step_vec3(Vec3 negative, Vec3 positive, float control) {
    __m128 positive_mask = _mm_cmpge_ps(_mm_set1_ps(control), _mm_setzero_ps());
    __m128 negative_mask = _mm_cmplt_ps(_mm_set1_ps(control), _mm_setzero_ps());
    return (Vec3) { _mm_or_ps(
        _mm_and_ps(positive_mask, positive.m),
        _mm_and_ps(negative_mask, negative.m)
    ) };
}

static inline float min_float(float a, float b) {
    __m128 ma = _mm_set_ss(a);
    __m128 mb = _mm_set_ss(b);
    return _mm_cvtss_f32(_mm_min_ps(ma, mb));
}

static inline float max_float(float a, float b) {
    __m128 ma = _mm_set_ss(a);
    __m128 mb = _mm_set_ss(b);
    return _mm_cvtss_f32(_mm_max_ps(ma, mb));
}

static inline float clamp_float(float x, float lower, float upper) {
    return min_float(upper, max_float(lower, x));
}

// True iff x, y, z are all real numbers (not NaN, not inf).
static inline int is_real(Vec3 a) {
    __m128 tmp = _mm_sub_ps(a.m, a.m); // Convert infs to NaNs.
    tmp = _mm_cmpunord_ps(tmp, tmp);
    return !(_mm_movemask_ps(tmp) & 7); // Only care about xyz, not extra lane.
}
#endif  // end SSE support check
#endif  // include guard

