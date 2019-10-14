/*  Supplemental double-precision 3-vector code that abstracts away a
 *  bunch of SIMD operations.  I don't plan to use this code elsewhere
 *  so I chose short simple names without worrying about name
 *  collisions.
 */
#ifndef JELLY_MCJELLYFACE_DVEC3_HPP
#define JELLY_MCJELLYFACE_DVEC3_HPP

#ifndef __AVX__
#error Need AVX support for dvec3.hpp
#else
#include "xmmintrin.h"
#include "immintrin.h"
#include "vec3.h"

struct Dvec3 {
    __m256d m;
    
    Dvec3() : m(_mm256_setzero_pd()) {}
    
    Dvec3(__m256d v) : m(v) {}
    
    Dvec3(double x, double y, double z) {
        m = _mm256_setr_pd(x, y, z, z);
    }
    
    // Dvec3(float, float, float) = delete;
    
    explicit Dvec3(Vec3 v) : m(_mm256_cvtps_pd(v.m)) {}
    
    explicit operator Vec3() const {
        return Vec3 { _mm256_cvtpd_ps(m) };
    }
};

static inline const double* as_doubles(const __m256d& v) {
    return reinterpret_cast<const double*>(&v);
}

static inline const double* as_doubles(const Dvec3& v) {
    return as_doubles(v.m);
}

static inline double X(const Dvec3& a) { return as_doubles(a)[0]; }
static inline double Y(const Dvec3& a) { return as_doubles(a)[1]; }
static inline double Z(const Dvec3& a) { return as_doubles(a)[2]; }

static inline __m256d all_x(const __m256d& m) {
    return _mm256_broadcast_sd(as_doubles(m) + 0);
}
static inline __m256d all_y(const __m256d& m) {
    return _mm256_broadcast_sd(as_doubles(m) + 1);
}
static inline __m256d all_z(const __m256d& m) {
    return _mm256_broadcast_sd(as_doubles(m) + 2);
}

static inline Dvec3 add(Dvec3 a, Dvec3 b) {
    return Dvec3 { _mm256_add_pd(a.m, b.m) };
}

static inline void iadd(Dvec3* a, Dvec3 b) {
    *a = add(*a, b);
}

inline Dvec3 operator+ (Dvec3 a, Dvec3 b) { return add(a, b); }
inline Dvec3& operator+= (Dvec3& a, Dvec3 b) { iadd(&a, b); return a; }

static inline Dvec3 sub(Dvec3 a, Dvec3 b) {
    return Dvec3 { _mm256_sub_pd(a.m, b.m) };
}

static inline void isub(Dvec3* a, Dvec3 b) {
    *a = sub(*a, b);
}

inline Dvec3 operator- (Dvec3 a, Dvec3 b) { return sub(a, b); }
inline Dvec3& operator-= (Dvec3& a, Dvec3 b) { isub(&a, b); return a; }

static inline Dvec3 scale(double c, Dvec3 a) {
    return Dvec3 { _mm256_mul_pd(_mm256_set1_pd(c), a.m) };
}

static inline void iscale(Dvec3* a, double c) {
    *a = scale(c, *a);
}

inline Dvec3 operator* (Dvec3 a, double c) { return scale(c, a); }
inline Dvec3 operator* (double c, Dvec3 a) { return scale(c, a); }
inline Dvec3& operator*= (Dvec3& a, double c) { iscale(&a, c); return a; }

static inline Dvec3 elementwise_mul(Dvec3 a, Dvec3 b) {
    return Dvec3 { _mm256_mul_pd(a.m, b.m) };
}

static inline double dot(Dvec3 a, Dvec3 b) {
    Dvec3 product = elementwise_mul(a, b);
    return X(product) + Y(product) + Z(product);
}

static inline Dvec3 cross(Dvec3 a, Dvec3 b) {
    double a_x = X(a);
    double a_y = Y(a);
    double a_z = Z(a);
    
    double b_x = X(b);
    double b_y = Y(b);
    double b_z = Z(b);
    
    Dvec3 a_yzx(a_y, a_z, a_x);
    Dvec3 a_zxy(a_z, a_x, a_y);
    Dvec3 b_yzx(b_y, b_z, b_x);
    Dvec3 b_zxy(b_z, b_x, b_y);
    
    return elementwise_mul(a_yzx, b_zxy) - elementwise_mul(a_zxy, b_yzx);
}

static inline __m256d all_sum_squares(Dvec3 a) {
    Dvec3 squares = elementwise_mul(a, a);
    __m256d xy = _mm256_add_pd(all_x(squares.m), all_y(squares.m));
    __m256d xyz = _mm256_add_pd(xy, all_z(squares.m));
    return xyz;
}

static inline double sum_squares(Dvec3 a) {
    return Z(all_sum_squares(a));
}

static inline double magnitude(Dvec3 a) {
    return sqrt(sum_squares(a));
}

// Mask zeroes out result in case of zero magnitude. (Also zeros out
// lanes of the result that were zero, but that is a no-op side
// effect)
static inline Dvec3 normalize(Dvec3 a, double* magnitude_out = nullptr) {
    double magnitude_ = magnitude(a);
    __m256d all_magnitude = _mm256_set1_pd(magnitude_);
    __m256d normalized256d = _mm256_div_pd(a.m, all_magnitude);
    __m256d mask = _mm256_cmp_pd(a.m, _mm256_setzero_pd(), _CMP_GT_OQ);
    if (magnitude_out) *magnitude_out = magnitude_;
    return Dvec3 { _mm256_and_pd(mask, normalized256d) };
}

// Return control >= 0 ? x : 0
static inline double step(double x, double control) {
    return control >= 0 ? x : 0;
}

// Return control >= 0 ? positive : negative
static inline Dvec3 step(Dvec3 negative, Dvec3 positive, double control) {
    __m256d control256d = _mm256_set1_pd(control);
    __m256d zero = _mm256_setzero_pd();
    __m256d mask = _mm256_cmp_pd(control256d, zero, _CMP_GT_OQ);
    return Dvec3 { _mm256_blendv_pd(negative.m, positive.m, mask) };
}

static inline double min_float(double a, double b) {
    return a < b ? a : b;
}

static inline double max_float(double a, double b) {
    return a > b ? a : b;
}

static inline double clamp_float(double x, double lower, double upper) {
    return min_float(upper, max_float(lower, x));
}

// True iff x, y, z are all real numbers (not NaN, not inf).
static inline int is_real(Dvec3 a) {
    Dvec3 z = a - a;
    return (X(z) == 0 && Y(z) == 0 && Z(z) == 0);
}

#endif  // end AVX support check
#endif  // include guard

