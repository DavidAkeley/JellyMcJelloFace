/*  Fallback code for no SIMD support computers (i.e. everyone using  a  web
 *  browser  right  now).  Fuck  you ECMA TC39, simd.js was a great idea and
 *  WebAssembly SIMD support is not an adequate replacement (yet)!
 *  (because it's not implemented, no shit it's inadequate).
 */
#include <math.h>
typedef struct vec3 { float x, y, z, pad; } Vec3;

static inline float X(Vec3 a) { return a.x; }
static inline float Y(Vec3 a) { return a.y; }
static inline float Z(Vec3 a) { return a.z; }

static inline Vec3 vec3(float x, float y, float z) {
    return (Vec3) { x, y, z, 0.0f };
}

static inline Vec3 add(Vec3 a, Vec3 b) {
    return (Vec3) { a.x+b.x, a.y+b.y, a.z+b.z, 0.0f };
}

static inline void iadd(Vec3* a, Vec3 b) {
    *a = add(*a, b);
}

static inline Vec3 sub(Vec3 a, Vec3 b) {
    return (Vec3) { a.x-b.x, a.y-b.y, a.z-b.z, 0.0f };
}

static inline void isub(Vec3* a, Vec3 b) {
    *a = sub(*a, b);
}

static inline Vec3 scale(float c, Vec3 a) {
    return (Vec3) { c*a.x, c*a.y, c*a.z, 0.0f };
}

static inline void iscale(Vec3* a, float c) {
    *a = scale(c, *a);
}

static inline Vec3 elementwise_mul(Vec3 a, Vec3 b) {
    return (Vec3) { a.x*b.x, a.y*b.y, a.z*b.z, 0.0f };
}

static inline float dot(Vec3 a, Vec3 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

static inline Vec3 cross(Vec3 a, Vec3 b) {
    return (Vec3) { a.y*b.z - a.z*b.y, b.x*a.z - a.x*b.z, a.x*b.y - a.y*b.x, 1.0f };
}

static inline float sum_squares(Vec3 a) {
    return dot(a, a);
}

static inline float magnitude(Vec3 a) {
    return sqrt(sum_squares(a));
}

static inline Vec3 normalize(Vec3 a) {
    return scale(1.0f/magnitude(a), a);
}

static inline Vec3 normalize_magnitude(Vec3 a, float* magnitude_output) {
    float m = magnitude(a);
    *magnitude_output = m;
    return scale(1.0f/m, a);
}

static inline float step(float x, float control) {
    return control >= 0 ? x : 0;
}

static inline Vec3 step_vec3(Vec3 negative, Vec3 positive, float control) {
    return control >= 0 ? positive : negative;
}

static inline float min_float(float a, float b) {
    return a < b ? a : b;
}

static inline float max_float(float a, float b) {
    return a > b ? a : b;
}

static inline float clamp_float(float x, float lower, float upper) {
    return min_float(upper, max_float(lower, x));
}

static inline int is_real(Vec3 a) {
    Vec3 zero = sub(a, a);
    return (zero.x == 0) & (zero.y == 0) & (zero.z == 0);
}

// If the profanity in the header offends you, then you really shouldn't be
// poking around in files with names like vec3-fallback.inc. Does that name
// give you a premonition that I was HAPPY when I wrote this code?

