/*
 * Copyright (C) 2024, Michael Steiner, Graz University of Technology.
 * This code is licensed under the MIT license.
 */

#pragma once

#include <filesystem>
#include <string>

#include "util/helper_math_extension.h"

// #define DEBUG_RAY make_float2(997, 204)
// #define DEBUG_RAY_IDX (DEBUG_RAY.x + 1559 * DEBUG_RAY.y)
#undef near
#undef far

// ------------------------------------------------------------------
// Camera Matrix
// ------------------------------------------------------------------

struct CameraMatrix
{
    float4 m0;
    float4 m1;
    float4 m2;

    inline __device__ __host__ float distance(const CameraMatrix& B) const
    {
        float4 tmp = fabs(m0 - B.m0);
        float dist = tmp.x + tmp.y + tmp.z + tmp.w;
        tmp = fabs(m1 - B.m1);
        dist += tmp.x + tmp.y + tmp.z + tmp.w;
        tmp = fabs(m2 - B.m2);
        dist += tmp.x + tmp.y + tmp.z + tmp.w;
        return dist;
    }

    inline __device__ __host__ float3 transform(const float4 x) const
    {
        return make_float3(dot(x, m0), dot(x, m1), dot(x, m2));
    }

    inline __device__ __host__ CameraMatrix inverse() const
    {
        float3 rot0_t{m0.x, m1.x, m2.x};
        float3 rot1_t{m0.y, m1.y, m2.y};
        float3 rot2_t{m0.z, m1.z, m2.z};

        float3 t{m0.w, m1.w, m2.w};
        return {
            make_float4(rot0_t, -dot(rot0_t, t)),
            make_float4(rot1_t, -dot(rot1_t, t)),
            make_float4(rot2_t, -dot(rot2_t, t)),
        };
    }

    inline __device__ __host__ float3 getTranslation() const
    {
        return make_float3(m0.w, m1.w, m2.w);
    }
};



enum ContractionType
{
    NONE = 0,
    AABB = 1,
    WARP_AABB_L2 = 2,
    WARP_AABB_LINF = 3
};

struct SceneInfo
{
    int grid_resolution;
    int grid_nlvl;
    int vals_per_lvl;
    bool normalized;

    float3 aabb_from;
    float3 aabb_to;

    float3 contraction_aabb_from;
    float3 contraction_aabb_to;
    ContractionType contraction_type;


    bool is_open_gl;
    float alpha_thre;
};

struct StepsizeInfo
{
    float cone_angle;
    float log1p_c;
    float stepsize;
    float near;
    float near_i;
    float far;

    float dt_min;
    float dt_max;
    float t_min;
    float t_max;
    float i_min;
    float i_max;

    void setAdditionalInfo(const SceneInfo &scene_info)
    {
        dt_min = stepsize;
        dt_max = fmaxf(fmaxf((scene_info.aabb_to.x - scene_info.aabb_from.x) / scene_info.grid_resolution * 1.732f,
                             (scene_info.aabb_to.y - scene_info.aabb_from.y) / scene_info.grid_resolution * 1.732f),
                             (scene_info.aabb_to.z - scene_info.aabb_from.z) / scene_info.grid_resolution * 1.732f);

        t_min = ((double) dt_min) / cone_angle;
        double d_i_min = 1.0 / cone_angle;
        i_min = d_i_min;

        double d_log1p_c = log(1.0 + cone_angle);
        log1p_c = d_log1p_c;

        double d_i_max = log(dt_max / (double) dt_min) / d_log1p_c + d_i_min;
        i_max = d_i_max;
        t_max = t_min * pow(1.0 + cone_angle, d_i_max - d_i_min);

        near_i = step_from_t(near);
    }

    void changeSamplingRate(const float factor, const SceneInfo& scene_info)
    {
        cone_angle /= factor;
        stepsize /= factor;
        setAdditionalInfo(scene_info);
    }

    inline __device__ __host__
    float t_from_step(const float step) const
    {
        if (cone_angle <= 0.0f)
            return step * dt_min;

        if (step >= i_max)
            return t_max + (step - i_max) * dt_max;
        else if (step < i_min)
            return step * dt_min;
        else
            return t_min * powf(1.f + cone_angle, step - i_min);
    }

    inline __device__ __host__
    float step_from_t(const float t) const
    {
        if (cone_angle <= 0.0f)
            return t / dt_min;

        if (t >= t_max)
            return (t - t_max) / dt_max + i_max;
        else if (t < t_min)
            return t / dt_min;
        else
            return logf(t / t_min) / log1p_c + i_min;
    }
};

struct CameraInfo
{
    int2 resolution;
    bool is_open_gl;

    float2 focal;
    float2 principal;

    CameraMatrix cam2world;
    CameraMatrix world2cam;

    float aperature = 0.0f;
    float focus_z = 1.0f;

    void resize(int2 new_resolution)
    {
        focal = new_resolution.y / (float) resolution.y * focal; // keep same fovy
        principal = principal * make_float2(new_resolution) / make_float2(resolution);
        resolution = new_resolution;
    }

    CameraInfo createUpdated(CameraMatrix& cam2world, int2 resolution) const
    {
        CameraInfo new_cam_info = *this;
        new_cam_info.cam2world = cam2world;
        new_cam_info.world2cam = cam2world.inverse();
        new_cam_info.resize(resolution);
        return new_cam_info;
    }
};

struct Segment
{
    float begin;
    float end;
    bool hit;
};

struct CameraPathDataEntry
{
    CameraMatrix transform_matrix;
    std::filesystem::path image_path;
};

struct RaymarchInfo
{
    CameraInfo cam_info;
    CameraInfo next_cam_info;
    StepsizeInfo stepsize_info;
    int sample_index;
    bool motion_blur;
    bool deterministic;
};


enum InterpolFunction
{
    Linear = 0,
    Smoothstep = 1,
    Smootherstep = 2,
    Nearest = 3
};