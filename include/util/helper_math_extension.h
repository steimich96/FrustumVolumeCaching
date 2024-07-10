/*
 * Copyright (C) 2024, Michael Steiner, Graz University of Technology.
 * This code is licensed under the MIT license.
 */

#pragma once

#include "helper_math.h"
#include "cuda_fp16.h"

// ------------------------------------------------------------------
// half 4
// ------------------------------------------------------------------

struct __builtin_align__(8) half4
{
    __half2 a;
    __half2 b;
};

// ------------------------------------------------------------------
// int divide
// ------------------------------------------------------------------

inline __host__ __device__ int3 operator/(int3 a, int3 b)
{
    return make_int3(a.x / b.x, a.y / b.y, a.z / b.z);
}
inline __host__ __device__ void operator/=(int3 &a, int3 b)
{
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
}
inline __host__ __device__ int3 operator/(int3 a, int b)
{
    return make_int3(a.x / b, a.y / b, a.z / b);
}
inline __host__ __device__ void operator/=(int3 &a, int b)
{
    a.x /= b;
    a.y /= b;
    a.z /= b;
}
inline __host__ __device__ int3 operator/(float b, int3 a)
{
    return make_int3(b / a.x, b / a.y, b / a.z);
}

inline __host__ __device__ int2 operator/(int2 a, int2 b)
{
    return make_int2(a.x / b.x, a.y / b.y);
}
inline __host__ __device__ void operator/=(int2 &a, int2 b)
{
    a.x /= b.x;
    a.y /= b.y;
}
inline __host__ __device__ int2 operator/(int2 a, int b)
{
    return make_int2(a.x / b, a.y / b);
}
inline __host__ __device__ void operator/=(int2 &a, int b)
{
    a.x /= b;
    a.y /= b;
}
inline __host__ __device__ int2 operator/(float b, int2 a)
{
    return make_int2(b / a.x, b / a.y);
}

// ------------------------------------------------------------------
// round (device only)
// ------------------------------------------------------------------

inline __device__ int3 make_int3_rn(const float3 a)
{
    return make_int3(__float2int_rn(a.x), __float2int_rn(a.y), __float2int_rn(a.z));
}
inline __device__ int3 make_int3_rd(const float3 a)
{
    return make_int3(__float2int_rd(a.x), __float2int_rd(a.y), __float2int_rd(a.z));
}
inline __device__ int3 make_int3_ru(const float3 a)
{
    return make_int3(__float2int_ru(a.x), __float2int_ru(a.y), __float2int_ru(a.z));
}

inline __device__ int2 make_int2_rn(const float2 a)
{
    return make_int2(__float2int_rn(a.x), __float2int_rn(a.y));
}
inline __device__ int2 make_int2_rd(const float2 a)
{
    return make_int2(__float2int_rd(a.x), __float2int_rd(a.y));
}
inline __device__ int2 make_int2_ru(const float2 a)
{
    return make_int2(__float2int_ru(a.x), __float2int_ru(a.y));
}

// ------------------------------------------------------------------
// dim3
// ------------------------------------------------------------------

inline __host__ __device__ dim3 toDim3(const uint x) { return dim3{x, x, x}; }
inline __host__ __device__ dim3 toDim3(const uint2 a) { return dim3{a.x, a.y, 1}; }
inline __host__ __device__ dim3 toDim3(const uint3 a) { return dim3{a.x, a.y, a.z}; }

inline __host__ __device__ uint divRoundUp(const uint a, const uint b)
{
    return (a + b - 1) / b;
}
inline __host__ __device__ uint3 divRoundUp(const int3 a, const int b)
{
    return make_uint3(divRoundUp(a.x, b), divRoundUp(a.y, b), divRoundUp(a.z, b));
}
inline __host__ __device__ uint2 divRoundUp(const int2 a, const int b)
{
    return make_uint2(divRoundUp(a.x, b), divRoundUp(a.y, b));
}


// ------------------------------------------------------------------
// Sigmoid
// ------------------------------------------------------------------

inline __host__ __device__ float sigmoid(const float x)
{
    return 1.0f / (1.0f + exp(-x));
}
inline __host__ __device__ float2 sigmoid(const float2 v)
{
    return make_float2(sigmoid(v.x), sigmoid(v.y));
}
inline __host__ __device__ float3 sigmoid(const float3 v)
{
    return make_float3(sigmoid(v.x), sigmoid(v.y), sigmoid(v.z));
}

// ------------------------------------------------------------------
// Smoothstep
// ------------------------------------------------------------------

inline __device__ __host__ float smoothstep(float x)
{
    return (x*x*(3.0f - (2.0f*x)));
}
inline __device__ __host__ float2 smoothstep(float2 x)
{
    return (x*x*(make_float2(3.0f) - (2.0f*x)));
}
inline __device__ __host__ float3 smoothstep(float3 x)
{
    return (x*x*(make_float3(3.0f) - (2.0f*x)));
}
inline __device__ __host__ float4 smoothstep(float4 x)
{
    return (x*x*(make_float4(3.0f) - (2.0f*x)));
}

// ------------------------------------------------------------------
// Smootherstep
// ------------------------------------------------------------------

inline __device__ __host__ float smootherstep(float x)
{
    return (x*x*x*(x*(6.0f * x - 15.0f) + 10.0f));
}
inline __device__ __host__ float2 smootherstep(float2 x)
{
    return (x*x*x*(x*(make_float2(6.0f) * x - 15.0f) + 10.0f));
}
inline __device__ __host__ float3 smootherstep(float3 x)
{
    return (x*x*x*(x*(make_float3(6.0f) * x - 15.0f) + 10.0f));
}
inline __device__ __host__ float4 smootherstep(float4 x)
{
    return (x*x*x*(x*(make_float4(6.0f) * x - 15.0f) + 10.0f));
}

// ------------------------------------------------------------------
// Mix
// ------------------------------------------------------------------

inline __device__ __host__ float3 mix(float3 a, float3 b, float t)
{
    return a + (b - a) * t;
}