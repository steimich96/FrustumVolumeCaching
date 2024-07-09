#pragma once

#include "common.h"

// Code adapted from https://github.com/NVlabs/tiny-cuda-nn/blob/6f018a9cd1b369bcb247e1d539968db8e48b2b3f/include/tiny-cuda-nn/vec.h

struct quat {
	quat() = default;
	__host__ __device__ quat(float w, float x, float y, float z) : w{w}, x{x}, y{y}, z{z} {}
	__host__ __device__ quat(const CameraMatrix& m) {
		// Code adapted from https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
		float tr = m.m0.x + m.m1.y + m.m2.z;

		if (tr > (float)0) {
			float S = sqrt(tr + (float)1) * (float)2; // S=4*qw
			w = (float)0.25 * S;
			x = (m.m2.y - m.m1.z) / S;
			y = (m.m0.z - m.m2.x) / S;
			z = (m.m1.x - m.m0.y) / S;
		} else if (m.m0.x > m.m1.y && m.m0.x > m.m2.z) {
			float S = sqrt((float)1 + m.m0.x - m.m1.y - m.m2.z) * (float)2; // S=4*x
			w = (m.m2.y - m.m1.z) / S;
			x = (float)0.25 * S;
			y = (m.m0.y + m.m1.x) / S;
			z = (m.m0.z + m.m2.x) / S;
		} else if (m.m1.y > m.m2.z) {
			float S = sqrt((float)1 + m.m1.y - m.m0.x - m.m2.z) * (float)2; // S=4*y
			w = (m.m0.z - m.m2.x) / S;
			x = (m.m0.y + m.m1.x) / S;
			y = (float)0.25 * S;
			z = (m.m1.z + m.m2.y) / S;
		} else {
			float S = sqrt((float)1 + m.m2.z - m.m0.x - m.m1.y) * (float)2; // S=4*z
			w = (m.m1.x - m.m0.y) / S;
			x = (m.m0.z + m.m2.x) / S;
			y = (m.m1.z + m.m2.y) / S;
			z = (float)0.25 * S;
		}
	}

	float w, x, y, z;
};

inline static __host__ __device__ quat operator-(const quat& a) { return {-a.w, -a.x, -a.y, -a.z}; }
inline static __host__ __device__ quat operator+(const quat& a, const quat& b) { return {a.w + b.w, a.x + b.x, a.y + b.y, a.z + b.z}; }
inline static __host__ __device__ quat operator-(const quat& a, const quat& b) { return {a.w - b.w, a.x - b.x, a.y - b.y, a.z - b.z}; }
inline static __host__ __device__ quat operator*(float a, const quat& b) { return {a * b.w, a * b.x, a * b.y, a * b.z}; }
inline static __host__ __device__ quat operator*(const quat& a, float b) { return {a.w * b, a.x * b, a.y * b, a.z * b}; }
inline static __host__ __device__ quat operator/(const quat& a, float b) { return {a.w / b, a.x / b, a.y / b, a.z / b}; }

inline static __host__ __device__ float dot(const quat& a, const quat& b) { return (a.w * b.w + a.x * b.x) + (a.y * b.y + a.z * b.z); }
inline static __host__ __device__ float length2(const quat& a) { return dot(a, a); }
inline static __host__ __device__ float length(const quat& a) { return sqrt(length2(a)); }

inline static __host__ __device__ quat mix(const quat& a, const quat& b, float t) { return a * ((float)1 - t) + b * t; }


inline static __host__ __device__ quat normalize(const quat& a) {
	float len = length(a);
	if (len <= (float)0) {
		return {(float)1, (float)0, (float)0, (float)0};
	}
	return a / len;
}


inline static __host__ __device__ quat cross(const quat& a, const quat& b) {
	return {
		a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
		a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
		a.w * b.y + a.y * b.w + a.z * b.x - a.x * b.z,
		a.w * b.z + a.z * b.w + a.x * b.y - a.y * b.x
	};
}


inline static __host__ __device__ quat slerp(const quat& x, const quat& y, float t) {
	quat z = y;

	float cos_theta = dot(x, y);

	// If cos_theta < 0, the interpolation will take the long way around the sphere.
	// To fix this, one quat must be negated.
	if (cos_theta < (float)0) {
		z = -y;
		cos_theta = -cos_theta;
	}

	// Perform a linear interpolation when cos_theta is close to 1 to avoid side effect of sin(angle) becoming a zero denominator
	if (cos_theta > (float)1 - std::numeric_limits<float>::epsilon()) {
		return mix(x, z, t);
	} else {
		// Essential Mathematics, page 467
		float angle = acos(cos_theta);
		return (sin((1.0f - t) * angle) * x + sin(t * angle) * z) / sin(angle);
	}
}


inline static __host__ __device__ float angle(const quat& x) {
	return acos(clamp(x.w, (float)-1, (float)1)) * (float)2;
}


inline static __host__ __device__ float3 axis(const quat& x) {
	const float tmp1 = (float)1 - x.w * x.w;
	if (tmp1 <= (float)0) {
		return {(float)0, (float)0, (float)1};
	}

	const float tmp2 = (float)1 / sqrt(tmp1);
	return {x.x * tmp2, x.y * tmp2, x.z * tmp2};
}


inline static __host__ __device__ CameraMatrix to_mat3(const quat& q) {
	float qxx = q.x * q.x, qyy = q.y * q.y, qzz = q.z * q.z;
	float qxz = q.x * q.z, qxy = q.x * q.y, qyz = q.y * q.z;
	float qwx = q.w * q.x, qwy = q.w * q.y, qwz = q.w * q.z;

    return {
        {(float)1 - (float)2 * (qyy +  qzz), (float)2 * (qxy - qwz), (float)2 * (qxz + qwy), 0.f},
        {(float)2 * (qxy + qwz), (float)1 - (float)2 * (qxx +  qzz), (float)2 * (qyz - qwx), 0.f},
        {(float)2 * (qxz - qwy), (float)2 * (qyz + qwx), (float)1 - (float)2 * (qxx +  qyy), 0.f},
    };
}


inline static __host__ __device__ CameraMatrix slerp(const CameraMatrix& a, CameraMatrix& b, float t) {
	return to_mat3(normalize(slerp(normalize(quat(a)), normalize(quat(b)), t)));
}


inline static __host__ __device__ float3 rotvec(const CameraMatrix& mat) {
	quat tmp = mat;
	return axis(tmp) * angle(tmp);
}