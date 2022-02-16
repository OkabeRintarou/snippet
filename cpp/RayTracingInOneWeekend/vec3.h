#pragma once

#include <cmath>
#include <cstdlib>

class vec3 {
public:
    vec3() { e[0] = e[1] = e[2] = .0f; }
    vec3(float e0, float e1, float e2) {
        e[0] = e0;
        e[1] = e1;
        e[2] = e2;
    }
    inline float x() const { return e[0]; }
    inline float y() const { return e[1]; }
    inline float z() const { return e[2]; }
    inline float r() const { return e[0]; }
    inline float g() const { return e[1]; }
    inline float b() const { return e[2]; }

    inline const vec3 &operator+() { return *this; }
    inline vec3 operator-() { return vec3(-e[0], -e[1], -e[2]); }
    inline float operator[](int i) const { return e[i]; }
    inline float &operator[](int i) { return e[i]; }

    inline float length() const { return sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]); }
    inline float squared_length() const { return e[0] * e[0] + e[1] * e[1] + e[2] * e[2]; }

private:
    float e[3];
};

inline vec3 operator+(const vec3 &lhs, const vec3 &rhs) {
    return vec3(lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2]);
}

inline vec3 operator-(const vec3 &lhs, const vec3 &rhs) {
    return vec3(lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2]);
}

inline vec3 operator*(const vec3 &lhs, const vec3 &rhs) {
    return vec3(lhs[0] * rhs[0], lhs[1] * rhs[1], lhs[2] * rhs[2]);
}

inline vec3 operator*(const vec3 &lhs, float v) {
    return vec3(lhs[0] * v, lhs[1] * v, lhs[2] * v);
}

inline vec3 operator/(const vec3 &lhs, const vec3 &rhs) {
    return vec3(lhs[0] / rhs[0], lhs[1] / rhs[1], lhs[2] / rhs[2]);
}
inline vec3 operator/(const vec3 &lhs, float v) {
    return vec3(lhs[0] / v, lhs[1] / v, lhs[2] / v);
}

inline float dot(const vec3 &lhs, const vec3 &rhs) {
    return lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2];
}

inline vec3 cross(const vec3 &lhs, const vec3 &rhs) {
    return vec3(lhs[1] * rhs[2] - lhs[2] * rhs[1],
                -(lhs[0] * rhs[2] - lhs[2] * rhs[0]),
                lhs[0] * rhs[1] - lhs[1] * rhs[0]);
}

inline vec3 unit_vector(const vec3 &v) {
    return v / v.length();
}