
#pragma once

#include "common.h"
#include "util/random_val.h"
#include "util/quat.h"
#include "util/helper_math_extension.h"

#undef near
#undef far

// ------------------------------------------------------------------
// AABB/GRID CONTAINS
// ------------------------------------------------------------------

template<typename T_VAL, typename T_RANGE>
inline __device__ __host__ bool inRange(const T_VAL val, const T_RANGE from_incl, const T_RANGE to_excl)
{
    return val >= from_incl && val < to_excl;
}

inline __device__ __host__ bool aabbContains(const float3 a, const float3 from, const float3 to)
{
    return inRange(a.x, from.x, to.x) && inRange(a.y, from.y, to.y) && inRange(a.z, from.z, to.z);
}

// float3
inline __device__ __host__ bool gridContainsXY(const float3 a, const int3 grid_dims)
{
    return inRange(a.x, 0, grid_dims.x) && inRange(a.y, 0, grid_dims.y);
}
inline __device__ __host__ bool gridContains3D(const float3 a, const int3 grid_dims)
{
    return inRange(a.x, 0, grid_dims.x) && inRange(a.y, 0, grid_dims.y) && inRange(a.z, 0, grid_dims.z);
}

// int3
inline __device__ __host__ bool gridContains3D(const int3 a, const int3 grid_dims)
{
    return inRange(a.x, 0, grid_dims.x) && inRange(a.y, 0, grid_dims.y) && inRange(a.z, 0, grid_dims.z);
}

// uint3, int
inline __device__ __host__ bool gridContains3D(const uint3 idx, const int3 grid_dims)
{
    return idx.x < grid_dims.x && idx.y < grid_dims.y && idx.z < grid_dims.z;
}
inline __device__ __host__ bool gridContains3D(const uint3 idx, const int grid_dim)
{
    return gridContains3D(idx, make_int3(grid_dim));
}


// ------------------------------------------------------------------
// MORTON ORDER (Z-ORDER)
// ------------------------------------------------------------------

__host__ __device__ inline uint32_t morton2D_invert(uint32_t x)
{
    x = x & 0x5555555555555555;
    x = (x | (x >> 1)) & 0x3333333333333333;
    x = (x | (x >> 2)) & 0x0F0F0F0F0F0F0F0F;
    x = (x | (x >> 4)) & 0x00FF00FF00FF00FF;
    x = (x | (x >> 8)) & 0x0000FFFF0000FFFF;
    x = (x | (x >> 16)) & 0x00000000FFFFFFFF;
    return x;
}
__host__ __device__ inline uint32_t expand_bits_2D(uint32_t x)
{
    x = (x | (x << 16)) & 0x0000FFFF0000FFFF;
    x = (x | (x << 8)) & 0x00FF00FF00FF00FF;
    x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0F;
    x = (x | (x << 2)) & 0x3333333333333333;
    x = (x | (x << 1)) & 0x5555555555555555;
    return x;
}
__host__ __device__ inline uint32_t morton2D(uint32_t x, uint32_t y)
{
    uint32_t xx = expand_bits_2D(x);
    uint32_t yy = expand_bits_2D(y);

    return xx | (yy << 1);
}


__host__ __device__ inline uint32_t morton3D_invert(uint32_t x) {
	x = x               & 0x49249249;
	x = (x | (x >> 2))  & 0xc30c30c3;
	x = (x | (x >> 4))  & 0x0f00f00f;
	x = (x | (x >> 8))  & 0xff0000ff;
	x = (x | (x >> 16)) & 0x0000ffff;
	return x;
}
__host__ __device__ inline uint32_t expand_bits_3D(uint32_t v) {
	v = (v * 0x00010001u) & 0xFF0000FFu;
	v = (v * 0x00000101u) & 0x0F00F00Fu;
	v = (v * 0x00000011u) & 0xC30C30C3u;
	v = (v * 0x00000005u) & 0x49249249u;
	return v;
}
__host__ __device__ inline uint32_t morton3D(uint32_t x, uint32_t y, uint32_t z) {
	uint32_t xx = expand_bits_3D(x);
	uint32_t yy = expand_bits_3D(y);
	uint32_t zz = expand_bits_3D(z);
	return xx | (yy << 1) | (zz << 2);
}


// ------------------------------------------------------------------
// CONVERSION: 3D <-> 1D
// ------------------------------------------------------------------

inline __device__ __host__ int to1D(const int3 coords3D, const int3 dims)
{
    return coords3D.x * dims.y * dims.z + coords3D.y * dims.z + coords3D.z;
}
inline __device__ __host__ int to1DMulti(const int3 coords3D, const int3 dims, const int lvl, const int vals_per_lvl)
{
    const int lvl_offset = vals_per_lvl * lvl;
    return to1D(coords3D, dims) + lvl_offset;
}
inline __device__ __host__ int to1DMulti(const int3 coords3D, const int dims, const int lvl, const int vals_per_lvl)
{
    return to1DMulti(coords3D, make_int3(dims), lvl, vals_per_lvl);
}

inline __device__ __host__ int2 to2D(const int index, const int2 dims)
{
    const int x = index / dims.y;
    const int y = index - x * dims.y;
    return make_int2(x, y);
}

inline __device__ __host__ int2 to2DFlipped(const int index, const int2 dims)
{
    const int y = index / dims.x;
    const int x = index - y * dims.x;
    return make_int2(x, y);
}

inline __device__ __host__ int3 to3D(const int index, const int3 dims)
{
    const int x = index / (dims.z * dims.y);
    const int y = (index - x * (dims.z * dims.y)) / dims.z;
    const int z = index - x * (dims.z * dims.y) - y * dims.z;
    return make_int3(x, y, z);
}

template <int BIT_WIDTH = 10>
inline __device__ __host__ int pack3D(const int3 index)
{
    static_assert(BIT_WIDTH <= 10, "Only 32 bits (max. 3*10) available");
    return (index.x << (2 * BIT_WIDTH)) | (index.y << (1 * BIT_WIDTH)) | index.z;
}

template <int BIT_WIDTH = 10>
inline __device__ __host__ int3 unpack3D(const int packed_idx)
{
    static_assert(BIT_WIDTH <= 10, "Only 32 bits (max. 3*10) available");
    constexpr int BIT_MASK = ((1 << BIT_WIDTH) - 1);
    return make_int3((packed_idx >> (2 * BIT_WIDTH)) & BIT_MASK, (packed_idx >> (1 * BIT_WIDTH)) & BIT_MASK, packed_idx & BIT_MASK);
}

inline __device__ __host__ float3 expand3D(const int3 sup_idx, const int brick_size, const float3 sub_idx, const int brick_padding = 0)
{
    return make_float3(sup_idx * (brick_size + 2 * brick_padding)) + sub_idx + brick_padding;
}
inline __device__ __host__ int3 expand3D(const int3 sup_idx, const int brick_size, const int3 sub_idx, const int brick_padding = 0)
{
    return sup_idx * (brick_size + 2 * brick_padding) + sub_idx + brick_padding;
}


// ------------------------------------------------------------------
// RAY GENERATION
// ------------------------------------------------------------------

struct FrustumRay
{
    float3 origin;
    float3 dir;

    inline __device__ __host__ float3 at(float t) const { return origin + t * dir; }
};

inline __device__ __host__
float3 generateRayDir(const float2 screen_point, const CameraInfo cam_info)
{
    float4 dir{
        (screen_point.x - cam_info.principal.x) / cam_info.focal.x,
        (screen_point.y - cam_info.principal.y) / cam_info.focal.y * (cam_info.is_open_gl ? -1.f : 1.f),
        (cam_info.is_open_gl ? -1.f : 1.f),
        0.f
    };

    return cam_info.cam2world.transform(dir);
}

inline __device__ __host__ CameraMatrix slerp(const CameraMatrix& a, const CameraMatrix& b, float t)
{
    return to_mat3(normalize(slerp(normalize(quat(a)), normalize(quat(b)), t)));
}

inline __device__ __host__ CameraMatrix camera_slerp(const CameraMatrix& a, const CameraMatrix& b, float t)
{
	CameraMatrix rot = slerp(a,b,t);
    auto trans = mix(float3{a.m0.w, a.m1.w, a.m2.w}, float3{b.m0.w, b.m1.w, b.m2.w}, t);
    rot.m0.w = trans.x;
    rot.m1.w = trans.y;
    rot.m2.w = trans.z;
	return rot;
};

inline __device__ __host__
FrustumRay generateRay(const int pixel_idx, const float2 screen_point, RaymarchInfo rm_info)
{
    CameraInfo& cam_info = rm_info.cam_info;

    if (!rm_info.deterministic && rm_info.motion_blur)
    {
        float t_motion = ld_random_val(rm_info.sample_index, (uint32_t) pixel_idx * 72239731);
        cam_info.cam2world = camera_slerp(cam_info.cam2world, rm_info.next_cam_info.cam2world, t_motion);
        cam_info.world2cam = cam_info.cam2world.inverse();
    }

    auto origin = cam_info.cam2world.getTranslation();
    auto dir = normalize(generateRayDir(screen_point, cam_info));

    if (!rm_info.deterministic && cam_info.aperature != 0.0f)
    {
        float3 lookat = origin + dir * cam_info.focus_z;
        float2 px_f = screen_point * float2{(float)cam_info.resolution.x, (float)cam_info.resolution.y};
        int2 px = int2{(int)px_f.x, (int)px_f.y};
        float2 blur = cam_info.aperature * square2disk_shirley(ld_random_val_2d(rm_info.sample_index, (uint32_t)px.x * 19349663 + (uint32_t)px.y * 96925573) * 2.0f - 1.0f);
        origin.x += cam_info.cam2world.m0.x * blur.x + cam_info.cam2world.m0.y * blur.y;
        origin.y += cam_info.cam2world.m1.x * blur.x + cam_info.cam2world.m1.y * blur.y;
        origin.z += cam_info.cam2world.m2.x * blur.x + cam_info.cam2world.m2.y * blur.y;
        dir = (lookat - origin) / cam_info.focus_z;
    }

    FrustumRay ray{
        origin,
        dir
    };

    return ray;
}

inline __device__ __host__
FrustumRay transformRay(const FrustumRay ray, const CameraMatrix mat)
{
    FrustumRay transformed_ray{
        mat.transform(make_float4(ray.origin, 1.0f)),
        mat.transform(make_float4(ray.dir, 0.0f)),
    };

    return transformed_ray;
}

inline __device__ __host__
float3 unit_to_01(const float3 dir)
{
    return (dir + 1.0f) / 2.0f;
}

// ------------------------------------------------------------------
// STEP
// ------------------------------------------------------------------

inline __device__ __host__
float calculate_stepsize(const float t, const float cone_angle, const float dt_min, const float dt_max)
{
    return clamp(t * cone_angle, dt_min, dt_max);
}

inline __device__ __host__
float calculate_stepsize(const float t, const StepsizeInfo stepsize_info)
{
    return calculate_stepsize(t, stepsize_info.cone_angle, stepsize_info.dt_min, stepsize_info.dt_max);
}

inline __device__ __host__
float valid_t_from_t(const float t, const float di, const StepsizeInfo info)
{
    // If t is the valid boundary, then this will give the first t0 with bin center t_mid > t
    return info.t_from_step(round(info.step_from_t(max(t, info.near)) - di) + di);
}
inline __device__ __host__
float valid_t_from_t(const float t, const StepsizeInfo info)
{
    return valid_t_from_t(t, info.near_i, info);
}

inline __device__ __host__
float max_step_in_scene(const StepsizeInfo stepsize_info, const SceneInfo scene_info)
{
    float scene_diag_length = length(scene_info.aabb_to - scene_info.aabb_from);
    return stepsize_info.step_from_t(scene_diag_length + stepsize_info.near);
}

inline __device__
int steps_in_segment(const Segment segment, const StepsizeInfo info)
{
    return __float2int_rn(info.step_from_t(segment.end)) - __float2int_rn(info.step_from_t(segment.begin));
}


// ------------------------------------------------------------------
// TRANSFORM
// ------------------------------------------------------------------

inline __device__ __host__
float3 cam2froxel(const float3 cam_point, const CameraInfo cam_info, const StepsizeInfo stepsize_info, float &t)
{
    float z = cam_point.z * (cam_info.is_open_gl ? -1.f : 1.f);
    t = length(cam_point);

    float3 froxel_point {
        x: cam_point.x / z * cam_info.focal.x + cam_info.principal.x,
        y: cam_point.y * (cam_info.is_open_gl ? -1.f : 1.f) / z * cam_info.focal.y + cam_info.principal.y,
        z: (stepsize_info.step_from_t(t) - stepsize_info.near_i) * (cam_point.z < 0.0f ? -1.0f : 1.0f)
    };

    return froxel_point;
}

inline __device__ __host__
float3 world2froxel(const float3 world_point, const CameraInfo cam_info, const StepsizeInfo stepsize_info, float &t)
{
    float3 cam_point = cam_info.world2cam.transform(make_float4(world_point, 1.0f));
    return cam2froxel(cam_point, cam_info, stepsize_info, t);
}

template <int BRICK_SIZE>
inline __device__ __host__
bool froxel2data(const float3 froxel_point, const char *froxel_brick_isset_array, const int *froxel_brick_index_array,
                 const int3 froxel_grid_bricks_per_dims, const int3 data_array_bricks_per_dim, bool use_inter_brick_interpolation,
                 float3 &data_array_point, bool &inside_brick)
{
    const int3 froxel_brick_idx = make_int3(froxel_point / BRICK_SIZE);
    float3 intra_brick_point = froxel_point - make_float3(froxel_brick_idx * BRICK_SIZE);

    if (use_inter_brick_interpolation)
    {
        inside_brick = aabbContains(intra_brick_point, make_float3(0.5f), make_float3(BRICK_SIZE - 0.5f));
    }
    else
    {
        inside_brick = true;
        intra_brick_point = clamp(intra_brick_point, 0.5f, BRICK_SIZE - 0.5f);
    }

    const int froxel_brick_idx1D = to1D(froxel_brick_idx, froxel_grid_bricks_per_dims);
    if (!froxel_brick_isset_array[froxel_brick_idx1D])
        return false;

    const int data_array_brick_idx1D = froxel_brick_index_array[froxel_brick_idx1D];
    const int3 data_array_brick_idx = to3D(data_array_brick_idx1D, data_array_bricks_per_dim);

    data_array_point = expand3D(data_array_brick_idx, BRICK_SIZE, intra_brick_point);
    return true;
}

template <typename T> inline __device__ T initValue();
template <> inline __device__ float initValue<float>() { return 0.0f; }
template <> inline __device__ float4 initValue<float4>() { return make_float4(0.0f); }

template <int BRICK_SIZE, typename T>
inline __device__ T fetchBrickTex3DPoint(cudaTextureObject_t tex_pt, float3 froxel_point,
    const char *froxel_brick_isset_array, const int *froxel_brick_index_array,
    const int3 froxel_grid_dims, const int3 froxel_grid_bricks_per_dims, const int3 data_array_bricks_per_dim)
{
    bool in_cache_fov = gridContainsXY(froxel_point, froxel_grid_dims);
    bool inside_grid = in_cache_fov && inRange(froxel_point.z, 0, froxel_grid_dims.z);

    T result = initValue<T>();

    bool inside_brick;
    float3 data_array_point;
    if (inside_grid && froxel2data<BRICK_SIZE>(froxel_point, froxel_brick_isset_array, froxel_brick_index_array, froxel_grid_bricks_per_dims,
                                               data_array_bricks_per_dim, true, data_array_point, inside_brick))
    {
        result = tex3D<T>(tex_pt, data_array_point);
    }

    return result;
}

inline __device__ __host__ float3 applyInterpolationFunction(const float3 t, const InterpolFunction interpol_fun)
{
    switch (interpol_fun)
    {
        case InterpolFunction::Smoothstep:   return smoothstep(t);
        case InterpolFunction::Smootherstep: return smootherstep(t);
        case InterpolFunction::Nearest:      return make_float3(nearbyintf(t.x), nearbyintf(t.y), nearbyintf(t.z));
        default:                             return t; //Linear
    }
}

template <typename T>
inline __device__ T lerp(T data[2][2][2], float3 t)
{
    return lerp(lerp(lerp(data[0][0][0], data[0][0][1], t.z),
                     lerp(data[0][1][0], data[0][1][1], t.z), t.y),
                lerp(lerp(data[1][0][0], data[1][0][1], t.z),
                     lerp(data[1][1][0], data[1][1][1], t.z), t.y), t.x);
}

inline __device__ float lerp(float data[2][2][2], float3 t)
{
    return d_lerp(d_lerp(d_lerp(data[0][0][0], data[0][0][1], t.z),
                     d_lerp(data[0][1][0], data[0][1][1], t.z), t.y),
                d_lerp(d_lerp(data[1][0][0], data[1][0][1], t.z),
                     d_lerp(data[1][1][0], data[1][1][1], t.z), t.y), t.x);
}

template <int BRICK_SIZE, typename T>
inline __device__ T fetchBrickTex3DInterpol(cudaTextureObject_t tex_pt, cudaTextureObject_t tex_linear, const float3 data_array_point, bool inside_brick,
    const float3 froxel_point, const char *froxel_brick_isset_array, const int *froxel_brick_index_array, bool use_inter_brick_interpolation,
    const int3 froxel_grid_dims, const int3 froxel_grid_bricks_per_dims, const int3 data_array_bricks_per_dim, const InterpolFunction interpol_fun)
{
    if (interpol_fun == InterpolFunction::Linear && (inside_brick || !use_inter_brick_interpolation))
        return tex3D<T>(tex_linear, data_array_point);

    if (interpol_fun == InterpolFunction::Nearest)
        return tex3D<T>(tex_pt, data_array_point);

    float3 froxel_idx = floorf(froxel_point - 0.5f);
    const float3 lerp_t = applyInterpolationFunction(froxel_point - 0.5f - froxel_idx, interpol_fun);
    float pos[3] = {lerp_t.z, lerp_t.y, lerp_t.x};

    T result = initValue<T>();
    for (uint32_t i = 0; i < 8; i++)
    {
        float weight = 1.0f;
        for (uint32_t dim = 0; dim < 3; dim++)
        {
            if ((i & (1 << dim)) == 0) {
                weight *= 1.0f - pos[dim];
            } else {
                weight *= pos[dim];
            }
        }

        const float3 xyz_offset = make_float3((i & (1 << 2)) >> 2, (i & (1 << 1)) >> 1, i & 1);
        T value = fetchBrickTex3DPoint<BRICK_SIZE, T>(tex_pt, froxel_idx + xyz_offset,
                                                      froxel_brick_isset_array, froxel_brick_index_array,
                                                      froxel_grid_dims, froxel_grid_bricks_per_dims, data_array_bricks_per_dim);

        result = weight * value + result;
    }

    return result;

    // T data[2][2][2];
    // for (int x = 0; x < 2; x++)
    //     for (int y = 0; y < 2; y++)
    //         for (int z = 0; z < 2; z++)
    //             data[x][y][z] = fetchBrickTex3DPoint<BRICK_SIZE, T>(tex_pt, froxel_idx + make_float3(x, y, z),
    //                                                                 froxel_brick_isset_array, froxel_brick_index_array,
    //                                                                 froxel_grid_dims, froxel_grid_bricks_per_dims, data_array_bricks_per_dim);
    
    // return lerp(data, lerp_t);
}


// ------------------------------------------------------------------
// RAY-BOX INTERSECTION
// ------------------------------------------------------------------

inline __host__ __device__ void _swap(float &a, float &b)
{
    float c = a;
    a = b;
    b = c;
}

inline __host__ __device__ bool ray_aabb_intersect(
    const FrustumRay ray,
    const float3 aabb_from,
    const float3 aabb_to,
    float& near,
    float& far)
{
    // aabb is [xmin, ymin, zmin, xmax, ymax, zmax]
    float tmin = (aabb_from.x - ray.origin.x) / ray.dir.x;
    float tmax = (aabb_to.x -   ray.origin.x) / ray.dir.x;
    if (tmin > tmax)
        _swap(tmin, tmax);

    float tymin = (aabb_from.y - ray.origin.y) / ray.dir.y;
    float tymax = (aabb_to.y -   ray.origin.y) / ray.dir.y;
    if (tymin > tymax)
        _swap(tymin, tymax);

    if (tmin > tymax || tymin > tmax)
    {
        near = 1e10;
        far = 1e10;
        return false;
    }

    if (tymin > tmin)
        tmin = tymin;
    if (tymax < tmax)
        tmax = tymax;

    float tzmin = (aabb_from.z - ray.origin.z) / ray.dir.z;
    float tzmax = (aabb_to.z   - ray.origin.z) / ray.dir.z;
    if (tzmin > tzmax)
        _swap(tzmin, tzmax);

    if (tmin > tzmax || tzmin > tmax)
    {
        near = 1e10;
        far = 1e10;
        return false;
    }

    if (tzmin > tmin)
        tmin = tzmin;
    if (tzmax < tmax)
        tmax = tzmax;

    near = tmin;
    far = tmax;
    return true;
}


// ------------------------------------------------------------------
// COLOR
// ------------------------------------------------------------------

inline __device__ __host__
float3 applyBackground(float3 color, const float transmittance, const float3 background_color)
{
    color.x = color.x + transmittance * background_color.x;
    color.y = color.y + transmittance * background_color.y;
    color.z = color.z + transmittance * background_color.z;
    return color;
}
inline __device__ __host__
float3 applyWhiteBackground(const float3 color, const float transmittance)
{
    const float3 white{1.0f, 1.0f, 1.0f};
    return applyBackground(color, transmittance, white);
}

inline __device__ __host__ uint to_8bit(const float f)
{
    return min(255, max(0, int(f * 256.f)));
}
inline __device__ __host__ uchar4 color_to_uchar4(const float3 color)
{
    uchar4 data;
    data.x = to_8bit(color.x);
    data.y = to_8bit(color.y);
    data.z = to_8bit(color.z);
    data.w = 255U;
    return data;
}
inline __device__ __host__ uint color_to_uint(const float3 color)
{
    return (to_8bit(color.x) << 0) +
      (to_8bit(color.y) << 8) +
      (to_8bit(color.z) << 16) +
      (255U << 24);
}
inline __device__ __host__ float3 uchar4_to_color(const uchar4 color)
{
    float3 data;
    data.x = (float)color.x / 255.f;
    data.y = (float)color.y / 255.f;
    data.z = (float)color.z / 255.f;

    return data;
}

// ------------------------------------------------------------------
// NORMS
// ------------------------------------------------------------------

inline __device__ __host__ float norm_l1(const float3 v)
{
    const float3 tmp = fabs(v);
    return tmp.x + tmp.y + tmp.z;
}

inline __device__ __host__ float norm_l2(const float3 v)
{
    return sqrtf(dot(v, v));
}

inline __device__ __host__ float norm_linf(const float3 v)
{
    const float3 tmp = fabs(v);
    return fmaxf(fmaxf(tmp.x, tmp.y), tmp.z);
}


// ------------------------------------------------------------------
// CONTRACTION
// ------------------------------------------------------------------

inline __device__ __host__
float3 contract_aabb(const float3 world_point, const float3 aabb_from, const float3 aabb_to)
{
    return (world_point - aabb_from) / (aabb_to - aabb_from);
}

inline __device__ __host__
float3 contract_norm(const float3 point_unit, const float norm)
{
    const float factor = fmaxf(norm, 1.0f); // do not contract points with norm <= 1

    float3 contracted_point = (2.0f - 1.0f / factor) * (point_unit / factor); // [-inf, inf] -> [-2, 2]
    return contracted_point / 4.0f + 0.5f; // [-2, 2] -> [0, 1]
}

inline __device__ __host__
float3 contract_linf(const float3 world_point, const float3 aabb_from, const float3 aabb_to)
{
    const float3 point_unit = contract_aabb(world_point, aabb_from, aabb_to) * 2.0f - 1.0; // [aabb_from, aabb_to] -> [-1, 1]
    return contract_norm(point_unit, norm_linf(point_unit));
}

inline __device__ __host__
float3 contract_l2(const float3 world_point, const float3 aabb_from, const float3 aabb_to)
{
    const float3 point_unit = contract_aabb(world_point, aabb_from, aabb_to) * 2.0f - 1.0; // [aabb_from, aabb_to] -> [-1, 1]
    return contract_norm(point_unit, norm_l2(point_unit));
}

inline __device__ __host__
float3 apply_contraction(const float3 world_point, const SceneInfo scene_info)
{
    switch (scene_info.contraction_type)
    {
        case ContractionType::AABB:
            return contract_aabb(world_point, scene_info.aabb_from, scene_info.aabb_to);
        case ContractionType::WARP_AABB_LINF:
            return contract_linf(world_point, scene_info.contraction_aabb_from, scene_info.contraction_aabb_to);
        case ContractionType::WARP_AABB_L2:
            return contract_l2(world_point, scene_info.contraction_aabb_from, scene_info.contraction_aabb_to);
        default:
            return world_point;
    }
}


// ------------------------------------------------------------------
// MULTIRESOLUTION GRID
// ------------------------------------------------------------------

inline __device__ __host__
float3 world_to_aabb_unit(const float3 world_point, const SceneInfo scene_info)
{
    // Scales world such that points inside innermost AABB are in [0, 1]
    float aabb_scale = 1 << (scene_info.grid_nlvl - 1);
    return (world_point - scene_info.aabb_from / aabb_scale) / ((scene_info.aabb_to - scene_info.aabb_from) / aabb_scale);
}

inline __device__ __host__
int mip_from_unit(const float3 point_unit)
{
    float3 scale = fabs(point_unit - 0.5f);
    float maxval = fmaxf(fmaxf(scale.x, scale.y), scale.z);

    int exponent;
    frexpf(maxval, &exponent);
    return max(0, exponent + 1);
}


template <int GRID_RESOLUTION, bool IS_MORTON=false>
inline __device__ __host__
bool grid_occupied_at(const float3 world_point, const SceneInfo scene_info, const uint8_t *grid_bitfield)
{
    const float3 point_unit = scene_info.normalized ? world_point : world_to_aabb_unit(world_point, scene_info);
    int mip = mip_from_unit(point_unit);

    if (mip >= scene_info.grid_nlvl)
        return false;

    const float3 point_unit_in_mip = (point_unit - 0.5f) * scalbnf(1.0f, -mip) + 0.5f;
    const int3 idx3D_in_mip = make_int3(point_unit_in_mip * GRID_RESOLUTION);
    return grid_occupied_at<GRID_RESOLUTION, IS_MORTON>(idx3D_in_mip, mip, grid_bitfield);
}

template <int GRID_RESOLUTION, bool IS_MORTON=false>
inline __device__ __host__
bool grid_occupied_at(int3 idx3D_in_mip, int mip, const uint8_t *grid_bitfield)
{
    if (IS_MORTON)
    {
        int idx = morton3D(idx3D_in_mip.x, idx3D_in_mip.y, idx3D_in_mip.z);
        int bitfield_idx = idx / 8 + (GRID_RESOLUTION * GRID_RESOLUTION * GRID_RESOLUTION * mip) / 8;
        return grid_bitfield[bitfield_idx] & (1 << (idx % 8));
    }
    else // linear
    {
        int idx = to1DMulti(idx3D_in_mip, GRID_RESOLUTION, mip, GRID_RESOLUTION * GRID_RESOLUTION * GRID_RESOLUTION);
        return grid_bitfield[idx / 8] & (1 << (idx % 8));
    }
}

template <int GRID_RESOLUTION>
inline __device__ __host__
bool grid_occupied_at(const float3 point, const float3 ray_dir, const SceneInfo scene_info, const uint8_t *grid_data, float& t_dist_to_next_voxel, int& idx)
{
    const float3 point_unit = scene_info.normalized ? point : world_to_aabb_unit(point, scene_info);
    int mip = mip_from_unit(point_unit);

    if (mip >= scene_info.grid_nlvl)
    {
        t_dist_to_next_voxel = 1e10f;
        idx = -1;
        return false;
    }

    const float3 point_unit_in_mip = (point_unit - 0.5f) * scalbnf(1.0f, -mip) + 0.5f;
    const int3 idx3D_in_mip = make_int3(point_unit_in_mip * GRID_RESOLUTION);

    const float3 mip_size = (scene_info.aabb_to - scene_info.aabb_from) * scalbnf(1.0f, -mip);
    const float3 voxel_size = mip_size / GRID_RESOLUTION;

    const int3 index_delta = make_int3(ray_dir.x > 0 ? 1 : 0, ray_dir.y > 0 ? 1 : 0, ray_dir.z > 0 ? 1 : 0);
    float3 t_dist = (make_float3(idx3D_in_mip + index_delta) / GRID_RESOLUTION - point_unit_in_mip) * voxel_size / ray_dir;
    t_dist = make_float3(
        (ray_dir.x == 0.0f) ? 1e10f : t_dist.x,
        (ray_dir.y == 0.0f) ? 1e10f : t_dist.y,
        (ray_dir.z == 0.0f) ? 1e10f : t_dist.z
    );
    t_dist_to_next_voxel = min(min(t_dist.x, t_dist.y), t_dist.z);

    idx = morton3D(idx3D_in_mip.x, idx3D_in_mip.y, idx3D_in_mip.z);
    int bitfield_idx = idx / 8 + (scene_info.vals_per_lvl * mip) / 8;
    return grid_data[bitfield_idx] & (1 << (idx % 8));
}

// ------------------------------------------------------------------
// Texture/Surface read/write
// ------------------------------------------------------------------

template <typename T>
inline __device__ T tex3D(cudaTextureObject_t tex, const float3 point)
{
    return tex3D<T>(tex, point.x, point.y, point.z);
}
template <typename T>
inline __device__ T tex3D(cudaTextureObject_t tex, const int3 point)
{
    return tex3D<T>(tex, point.x, point.y, point.z);
}

template <typename T>
inline __device__ void surf3Dwrite(T val, cudaSurfaceObject_t surf, const int3 idx)
{
    surf3Dwrite(val, surf, idx.x * sizeof(T), idx.y, idx.z);
}