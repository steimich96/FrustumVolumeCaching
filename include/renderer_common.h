/*
 * Copyright (C) 2024, Michael Steiner, Graz University of Technology.
 * This code is licensed under the MIT license.
 */

#pragma once

#include "cuda_runtime.h"

constexpr int DEFAULT_GRID_RESOLUTION = 128;
constexpr int GRID_IS_MORTON = true;


struct RayPayload // old RayPayload, with combined init, payload and result 
{
    float3 origin;
    float3 dir;
    int ray_id;
    float far;

    bool alive;
    float curr_t1;
    int subiter_n_steps_resample;
    int subiter_n_steps_cache;

    float total_contrib_resample;
    float total_contrib_cache;
    int total_n_steps_resample;
    int total_n_steps_cache;
    float3 rgb;
    float transmittance;
    float termination_depth;
    bool cache_overflow;
};

struct BlockPayloadCacheInit
{
    int curr_block_step;
    int n_alive_interior_rays;
    int n_total_bricks;
};

struct RayPayloadCacheInit
{
    float3 origin;
    float3 dir;
    float far;

    int subiter_n_steps;
    int subiter_offset_in_block;
    int total_n_steps;

    bool alive;
    float transmittance;
};


struct MissParams
{
};


// Ray Info - BASE

struct __align__(32) BaseRayInitInfo
{
    float3 origin;
    float3 dir;
    int ray_id;
    float far;
    //float near_i; needed if using SAMPLE_WITH_RTX
};

struct __align__(16) BaseRayResult
{
    float3 rgb;
    float transmittance;
};

struct BaseRayPayload
{
    float curr_t1;
};

struct SegmentedRayPayload
{
    float curr_t1;
    float curr_segment_end;
};