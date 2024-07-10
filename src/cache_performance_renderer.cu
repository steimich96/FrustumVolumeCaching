/*
 * Copyright (C) 2024, Michael Steiner, Graz University of Technology.
 * This code is licensed under the MIT license.
 */

#include "cache_performance_renderer.h"

#include "raymarch_common.h"
#include "util/cub_helper.h"

#ifdef RTX_ENABLED
#include "mesher.h"
#include "util/owl_helper.h"
#endif

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

using namespace cooperative_groups;
namespace cg = cooperative_groups;

template <bool FIRST_LATENT_IS_DENSITY = false>
__global__ void sample_cache_performance_kernel(int n_alive,
                                                int n_steps_curr_subiter,

                                                const RaymarchInfo rm_info,
                                                const RaymarchInfo cache_rm_info,
                                                const SceneInfo scene_info,
                                                const CachePerformanceRenderer::RenderMode render_mode,

                                                const uint8_t *__restrict__ occupancy_grid,
                                                const BaseRayInitInfo *__restrict__ ray_inits,
                                                SegmentedRayPayload *__restrict__ payloads,
                                                int *__restrict__ alive_buffer,

                                                CachePerformanceRenderer::SubiterRayEntryCumsum *__restrict__ subiter_ray_entries,

                                                cudaTextureObject_t known_ranges_to,
                                                const char *__restrict__ froxel_brick_isset_array,
                                                const int *__restrict__ froxel_brick_index_array,
                                                cudaTextureObject_t froxel_packed_brick_index_array_3D,
                                                const int3 froxel_grid_bricks_per_dims,
                                                const int3 froxel_grid_dims,

                                                cudaTextureObject_t data_isset_tex_pt,
                                                Cache::Textures data_array_textures,
                                                const int3 data_array_bricks_per_dim,

                                                float *__restrict__ sample_cache_t0,
                                                CachePerformanceRenderer::SampleCacheEntry *__restrict__ sample_cache_tmp,
                                                float *__restrict__ sample_resample_t0)
{
    const int curr_subiter_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (curr_subiter_idx >= n_alive)
        return;

    int step_resample = 0;
    int step_cache = 0;
    const BaseRayInitInfo ray_init = ray_inits[curr_subiter_idx];

    FrustumRay ray_world{ray_init.origin, ray_init.dir};
    FrustumRay ray_cache = transformRay(ray_world, cache_rm_info.cam_info.world2cam);

    // bool debug = false;
    // bool debug_ray = debug && ray_init.ray_id == (400 * rm_info.cam_info.resolution.x + 400);

    bool alive = true;
    SegmentedRayPayload payload = payloads[curr_subiter_idx];
    float t1 = payload.curr_t1;

    int steps_taken = 0;
    int step_cutoff = n_alive < 128 ? 1024 : 128; // If single block left, then try to get it done; else stop early
    while ((step_resample < n_steps_curr_subiter && step_cache < n_steps_curr_subiter) && steps_taken < step_cutoff)
    {
        steps_taken++;

        float t0 = t1;
        float dt = calculate_stepsize(t0, rm_info.stepsize_info);
        t1 = t0 + dt;
        float t_mid = (t0 + t1) * 0.5f;

        if (t_mid > ray_init.far)
        {
            alive = false;
            break;
        }

        if (CachePerformanceRenderer::SAMPLE_WITH_RTX && t_mid > payload.curr_segment_end)
            break;

        const float3 world_point = ray_init.origin + t_mid * ray_init.dir;

        if (CachePerformanceRenderer::SAMPLE_WITH_RTX || grid_occupied_at<DEFAULT_GRID_RESOLUTION, GRID_IS_MORTON>(world_point, scene_info, occupancy_grid))
        {
            float t_froxel;
            const float3 cache_cam_point = ray_cache.at(t_mid);
            const float3 froxel_point = cam2froxel(cache_cam_point, cache_rm_info.cam_info, cache_rm_info.stepsize_info, t_froxel);

            bool in_cache_fov = gridContainsXY(froxel_point, froxel_grid_dims);
            bool inside_grid = in_cache_fov && inRange(froxel_point.z, 0, froxel_grid_dims.z);

            bool in_unsampled_region = true;
            if (in_cache_fov)
            {
                float2 cache_pixel_known_range = make_float2(cache_rm_info.stepsize_info.near, tex2D<float>(known_ranges_to, froxel_point.x, froxel_point.y));
                in_unsampled_region = t_froxel < cache_pixel_known_range.x || t_froxel > cache_pixel_known_range.y;

                // if (t_froxel > cache_pixel_known_range.y && inside_grid)
                // {
                //     alive = false;
                //     break;
                // }
            }

            if (!inside_grid || in_unsampled_region)
            {
                if (render_mode == CachePerformanceRenderer::RenderMode::ONLY_CACHE)
                    continue;
                int row_offset = step_resample * n_alive + curr_subiter_idx;
                sample_resample_t0[row_offset] = t0;
                step_resample++;
            }
            else
            {
                if (render_mode == CachePerformanceRenderer::RenderMode::ONLY_RESAMPLED)
                    continue;
                const int3 froxel_brick_idx = make_int3(froxel_point / Cache::BRICK_SIZE);
                const int froxel_brick_idx1D = to1D(froxel_brick_idx, froxel_grid_bricks_per_dims);

                int packed_brick_index3D = tex3D<int>(froxel_packed_brick_index_array_3D, froxel_brick_idx);
                bool brick_isset = packed_brick_index3D >= 0;

                float3 intra_brick_point = froxel_point - make_float3(froxel_brick_idx * Cache::BRICK_SIZE);
                if (brick_isset)
                {
                    const int3 data_array_brick_idx = unpack3D(packed_brick_index3D);
                    float3 data_array_point = expand3D(data_array_brick_idx, Cache::BRICK_SIZE, intra_brick_point, Cache::BRICK_PADDING);

                    if (Cache::BRICK_PADDING > 0)
                    {
                        float density_interpol = tex3D<float>(data_array_textures.data_density_alpha_tex_linear, data_array_point);
                        float alpha = clamp(1.0f - exp(-density_interpol * dt), 0.0f, 1.0f);
                        if (alpha > scene_info.alpha_thre)
                        {
                            int row_offset = step_cache * n_alive + curr_subiter_idx;
                            sample_cache_tmp[row_offset] = {t0, packed_brick_index3D, density_interpol};
                            step_cache++;
                        }
                    }
                    else
                    {
                        float density_pt = tex3D<float>(data_array_textures.data_density_alpha_tex_pt, data_array_point);
                        if (density_pt > 0.0f) // Nearest-neighbor density. Can only be used as "isset" indicator
                        {
                            int row_offset = step_cache * n_alive + curr_subiter_idx;
                            sample_cache_t0[row_offset] = t0;

                            step_cache++;
                        }
                    }
                }
            }
        }
    }

    alive_buffer[curr_subiter_idx] = alive;

    subiter_ray_entries[curr_subiter_idx] = {
        /*has_resample: */ step_resample > 0,
        /*n_samples_resample: */ step_resample,
        /*has_cache: */ step_cache > 0,
        /*n_samples_cache: */ step_cache,
    };

    payload.curr_t1 = t1;
    payloads[curr_subiter_idx] = payload;
}
__global__ void prepareNetworkGeneration_perf_kernel(int n_alive,
                                                     const CachePerformanceRenderer::SubiterRayEntryCumsum *__restrict__ subiter_ray_entries_cumsum,
                                                     int *__restrict__ subiter_compact_indices_resample,
                                                     int *__restrict__ subiter_compact_indices_cache)
{
    const int curr_subiter_idx = (threadIdx.x + blockIdx.x * blockDim.x);
    if (curr_subiter_idx >= n_alive)
        return;

    const auto ray_entry_cumsum = subiter_ray_entries_cumsum[curr_subiter_idx];
    const auto ray_entry_cumsum_prev = curr_subiter_idx > 0 ? subiter_ray_entries_cumsum[curr_subiter_idx - 1] : CachePerformanceRenderer::SubiterRayEntryCumsum{0, 0, 0, 0};
    const auto ray_entry = CachePerformanceRenderer::SubiterRayEntry::createFromDiff(ray_entry_cumsum, ray_entry_cumsum_prev);

    if (ray_entry.has_resample)
        subiter_compact_indices_resample[ray_entry_cumsum.offset_resample - 1] = curr_subiter_idx;

    if (ray_entry.has_cache)
        subiter_compact_indices_cache[ray_entry_cumsum.offset_cache - 1] = curr_subiter_idx;
}

__global__ void generateNetworkInput_resample_perf_kernel(int n_threads,
                                                          int n_alive,
                                                          int n_samples_resample,
                                                          int n_steps_curr_subiter,
                                                          const RaymarchInfo rm_info,
                                                          const SceneInfo scene_info,
                                                          const BaseRayInitInfo *__restrict__ ray_inits,
                                                          const CachePerformanceRenderer::SubiterRayEntryCumsum *__restrict__ subiter_ray_entries_cumsum,
                                                          const int *__restrict__ subiter_compact_indices_resample,
                                                          const float *__restrict__ sample_t0_in,
                                                          float *__restrict__ sample_t0_out,
                                                          float3 *__restrict__ sample_pos,
                                                          float3 *__restrict__ sample_dir)
{
    const int idx = (threadIdx.x + blockIdx.x * blockDim.x);
    if (idx >= n_threads)
        return;

    const int curr_subiter_idx = subiter_compact_indices_resample[idx / n_steps_curr_subiter];
    const int step = idx % n_steps_curr_subiter;

    const auto ray_entry_cumsum = subiter_ray_entries_cumsum[curr_subiter_idx];
    const auto ray_entry_cumsum_prev = curr_subiter_idx > 0 ? subiter_ray_entries_cumsum[curr_subiter_idx - 1] : CachePerformanceRenderer::SubiterRayEntryCumsum{0, 0, 0, 0};
    const auto ray_entry = CachePerformanceRenderer::SubiterRayEntry::createFromDiff(ray_entry_cumsum, ray_entry_cumsum_prev);

    if (step >= ray_entry.n_samples_resample)
        return;

    const BaseRayInitInfo ray_init = ray_inits[curr_subiter_idx];
    int offset = ray_entry_cumsum.sample_offset_resample - ray_entry.n_samples_resample;

    const FrustumRay ray{ray_init.origin, ray_init.dir};

    int input_row_offset = step * n_alive + curr_subiter_idx;
    float t0 = sample_t0_in[input_row_offset];
    float dt = calculate_stepsize(t0, rm_info.stepsize_info);
    float t_mid = t0 + dt * 0.5f;

    int output_offset = offset + step;
    sample_t0_out[output_offset] = t0;
    sample_pos[output_offset] = apply_contraction(ray.at(t_mid), scene_info);
    sample_dir[output_offset] = unit_to_01(ray_init.dir);
}

template <bool FIRST_LATENT_IS_DENSITY = false>
__global__ void generateNetworkInput_preparation_perf_kernel(int n_threads,
                                                             int n_rays_cache,
                                                             int n_alive,
                                                             int n_samples_cache,
                                                             int n_steps_curr_subiter,

                                                             const RaymarchInfo rm_info,
                                                             const RaymarchInfo cache_rm_info,

                                                             const SceneInfo scene_info,
                                                             const BaseRayInitInfo *__restrict__ ray_inits,
                                                             const CachePerformanceRenderer::SubiterRayEntryCumsum *__restrict__ subiter_ray_entries_cumsum,
                                                             const int *__restrict__ subiter_compact_indices_cache,

                                                             const float *__restrict__ sample_t0_in,
                                                             float *__restrict__ sample_t0_out,
                                                             float3 *__restrict__ sample_froxel_pos,
                                                             tcnn::MatrixView<float> sample_pos,
                                                             tcnn::MatrixView<float> sample_dir,
                                                             tcnn::MatrixView<float> sample_init_viewdir)
{
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n_threads)
        return;

    int curr_subiter_idx = subiter_compact_indices_cache[idx / n_steps_curr_subiter];
    int step = idx % n_steps_curr_subiter;

    const auto ray_entry_cumsum = subiter_ray_entries_cumsum[curr_subiter_idx];
    const auto ray_entry_cumsum_prev = curr_subiter_idx > 0 ? subiter_ray_entries_cumsum[curr_subiter_idx - 1] : CachePerformanceRenderer::SubiterRayEntryCumsum{0, 0, 0, 0};
    const auto ray_entry = CachePerformanceRenderer::SubiterRayEntry::createFromDiff(ray_entry_cumsum, ray_entry_cumsum_prev);
    if (step >= ray_entry.n_samples_cache)
        return;

    int offset = ray_entry_cumsum.sample_offset_cache - ray_entry.n_samples_cache;

    const BaseRayInitInfo ray_init = ray_inits[curr_subiter_idx];

    const FrustumRay ray_world{ray_init.origin, ray_init.dir};
    const FrustumRay ray_cache = transformRay(ray_world, cache_rm_info.cam_info.world2cam);

    int input_row_offset = step * n_alive + curr_subiter_idx;
    float t0 = sample_t0_in[input_row_offset];
    float dt = calculate_stepsize(t0, rm_info.stepsize_info);
    float t_mid = t0 + dt * 0.5f;

    float froxel_t;
    const float3 world_point = ray_world.at(t_mid);
    const float3 froxel_point = cam2froxel(ray_cache.at(t_mid), cache_rm_info.cam_info, cache_rm_info.stepsize_info, froxel_t);
    const float3 cache_viewdir = normalize(world_point - cache_rm_info.cam_info.cam2world.getTranslation());

    int output_offset = offset + step;

    sample_t0_out[output_offset] = t0;
    sample_froxel_pos[output_offset] = froxel_point;

    const float3 sample_pos_tmp = apply_contraction(world_point, scene_info);
    sample_pos(0, output_offset) = sample_pos_tmp.x;
    sample_pos(1, output_offset) = sample_pos_tmp.y;
    sample_pos(2, output_offset) = sample_pos_tmp.z;

    float3 sample_dir_tmp = unit_to_01(ray_init.dir);
    sample_dir(0, output_offset) = sample_dir_tmp.x;
    sample_dir(1, output_offset) = sample_dir_tmp.y;
    sample_dir(2, output_offset) = sample_dir_tmp.z;

    float3 sample_init_viewdir_tmp = unit_to_01(cache_viewdir);
    sample_init_viewdir(0, output_offset) = sample_init_viewdir_tmp.x;
    sample_init_viewdir(1, output_offset) = sample_init_viewdir_tmp.y;
    sample_init_viewdir(2, output_offset) = sample_init_viewdir_tmp.z;
}

void inline __device__ setNetworkIntermediates_perf(half *network_input, const float4 vals, int row_offset, int batch_size, int i, int offset = 0)
{
    network_input[row_offset + (i * 4 + 0 + offset) * batch_size] = __float2half(vals.x);
    network_input[row_offset + (i * 4 + 1 + offset) * batch_size] = __float2half(vals.y);
    network_input[row_offset + (i * 4 + 2 + offset) * batch_size] = __float2half(vals.z);
    network_input[row_offset + (i * 4 + 3 + offset) * batch_size] = __float2half(vals.w);
}

template <bool FIRST_LATENT_IS_DENSITY = false>
__global__ void generateNetworkInput_cache_padded_kernel(int n_threads,
                                                         int n_rays_cache,
                                                         int n_alive,
                                                         int n_samples_cache,
                                                         int n_steps_curr_subiter,

                                                         const RaymarchInfo rm_info,
                                                         const RaymarchInfo cache_rm_info,

                                                         const SceneInfo scene_info,
                                                         const BaseRayInitInfo *__restrict__ ray_inits,
                                                         const CachePerformanceRenderer::SubiterRayEntryCumsum *__restrict__ subiter_ray_entries_cumsum,
                                                         const int *__restrict__ subiter_compact_indices_cache,

                                                         const CachePerformanceRenderer::SampleCacheEntry *__restrict__ sample_in,
                                                         float *__restrict__ sample_t0_out,
                                                         float3 *__restrict__ sample_froxel_pos,
                                                         tcnn::MatrixView<float> sample_pos,
                                                         tcnn::MatrixView<float> sample_dir,
                                                         tcnn::MatrixView<float> sample_init_viewdir,

                                                         int batch_size,
                                                         cudaTextureObject_t froxel_packed_brick_index_array_3D,
                                                         const int3 froxel_grid_bricks_per_dims,
                                                         const int3 froxel_grid_dims,

                                                         const int3 data_array_bricks_per_dim,
                                                         Cache::Textures data_array_textures,

                                                         half *__restrict__ mlp_head_network_inputs)
{
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n_threads)
        return;

    int curr_subiter_idx = subiter_compact_indices_cache[idx / n_steps_curr_subiter];
    int step = idx % n_steps_curr_subiter;

    const auto ray_entry_cumsum = subiter_ray_entries_cumsum[curr_subiter_idx];
    const auto ray_entry_cumsum_prev = curr_subiter_idx > 0 ? subiter_ray_entries_cumsum[curr_subiter_idx - 1] : CachePerformanceRenderer::SubiterRayEntryCumsum{0, 0, 0, 0};
    const auto ray_entry = CachePerformanceRenderer::SubiterRayEntry::createFromDiff(ray_entry_cumsum, ray_entry_cumsum_prev);
    if (step >= ray_entry.n_samples_cache)
        return;

    int offset = ray_entry_cumsum.sample_offset_cache - ray_entry.n_samples_cache;

    const BaseRayInitInfo ray_init = ray_inits[curr_subiter_idx];

    const FrustumRay ray_world{ray_init.origin, ray_init.dir};
    const FrustumRay ray_cache = transformRay(ray_world, cache_rm_info.cam_info.world2cam);

    int input_row_offset = step * n_alive + curr_subiter_idx;
    const CachePerformanceRenderer::SampleCacheEntry sample = sample_in[input_row_offset];
    float dt = calculate_stepsize(sample.t0, rm_info.stepsize_info);
    float t_mid = sample.t0 + dt * 0.5f;

    float froxel_t;
    const float3 world_point = ray_world.at(t_mid);
    const float3 froxel_point = cam2froxel(ray_cache.at(t_mid), cache_rm_info.cam_info, cache_rm_info.stepsize_info, froxel_t);
    const float3 cache_viewdir = normalize(world_point - cache_rm_info.cam_info.cam2world.getTranslation());

    int output_offset = offset + step;

    sample_t0_out[output_offset] = sample.t0;
    sample_froxel_pos[output_offset] = froxel_point;

    float3 sample_pos_tmp = apply_contraction(world_point, scene_info);
    sample_pos(0, output_offset) = sample_pos_tmp.x;
    sample_pos(1, output_offset) = sample_pos_tmp.y;
    sample_pos(2, output_offset) = sample_pos_tmp.z;

    float3 sample_dir_tmp = unit_to_01(ray_init.dir);
    sample_dir(0, output_offset) = sample_dir_tmp.x;
    sample_dir(1, output_offset) = sample_dir_tmp.y;
    sample_dir(2, output_offset) = sample_dir_tmp.z;

    float3 sample_init_viewdir_tmp = unit_to_01(cache_viewdir);
    sample_init_viewdir(0, output_offset) = sample_init_viewdir_tmp.x;
    sample_init_viewdir(1, output_offset) = sample_init_viewdir_tmp.y;
    sample_init_viewdir(2, output_offset) = sample_init_viewdir_tmp.z;

    const int3 froxel_brick_idx = make_int3(froxel_point / Cache::BRICK_SIZE);
    const float3 intra_brick_point = froxel_point - make_float3(froxel_brick_idx * Cache::BRICK_SIZE);

    float3 data_array_point = expand3D(unpack3D(sample.packed_brick_index3D), Cache::BRICK_SIZE, intra_brick_point, Cache::BRICK_PADDING);

    float weight_interpol = tex3D<float>(data_array_textures.data_isset_tex_linear, data_array_point);
    float density_interpol = sample.density_interpol;

    if (!FIRST_LATENT_IS_DENSITY)
        mlp_head_network_inputs[output_offset] = log(max(density_interpol, 1e-6f));

    for (int i = 0; i < Cache::N_DATA_ARRAYS; i++)
    {
        float4 data_interpol = tex3D<float4>(data_array_textures.data_tex_linear[i], data_array_point);

        if (i == 0 && FIRST_LATENT_IS_DENSITY)
        {
            float density = data_interpol.x;
            float alpha = clamp(1.0f - exp(-density * dt), 0.0f, 1.0f);
            data_interpol.x = log(max(density, 1e-6f)) * weight_interpol;
        }

        data_interpol /= weight_interpol;

        setNetworkIntermediates_perf(mlp_head_network_inputs, data_interpol, output_offset, batch_size, i, !FIRST_LATENT_IS_DENSITY);
    }
}

__global__ void generateNetworkInput_cache_perf_kernel(int n_samples_cache,
                                                       int batch_size,

                                                       const RaymarchInfo rm_info,
                                                       const CacheSettings cache_settings,

                                                       const cudaTextureObject_t known_ranges_to,
                                                       const char *froxel_brick_isset_array,
                                                       const int *froxel_brick_index_array,
                                                       cudaTextureObject_t froxel_packed_brick_index_array_3D,
                                                       const int3 froxel_grid_bricks_per_dims,
                                                       const int3 froxel_grid_dims,

                                                       const int3 data_array_bricks_per_dim,
                                                       Cache::Textures data_array_textures,

                                                       const float *__restrict__ sample_t0,
                                                       const float3 *__restrict__ sample_pos,
                                                       half *__restrict__ mlp_head_network_inputs)
{
    const int sample_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (sample_idx >= n_samples_cache)
        return;

    // static constexpr CacheSettings cache_settings = CachePerformanceRenderer::default_cache_settings;

    const float3 froxel_point = sample_pos[sample_idx];
    float3 froxel_idx = floorf(froxel_point - 0.5f);

    // const int3 froxel_brick_idx = make_int3(froxel_point / Cache::BRICK_SIZE);
    // const float3 intra_brick_point = clamp(froxel_point - make_float3(froxel_brick_idx * Cache::BRICK_SIZE), 0.5f, Cache::BRICK_SIZE - 0.5f);

    // const int froxel_brick_idx1D = to1D(froxel_brick_idx, froxel_grid_bricks_per_dims);

    // const int data_array_brick_idx1D = froxel_brick_index_array[froxel_brick_idx1D];
    // const int3 data_array_brick_idx = to3D(data_array_brick_idx1D, data_array_bricks_per_dim);

    // float3 data_array_point = expand3D(data_array_brick_idx, Cache::BRICK_SIZE, intra_brick_point);

    // float weight = tex3D<float>(data_array_textures.data_isset_tex_linear, data_array_point);

    // const float t0 = sample_t0[sample_idx];
    // const float dt = calculate_stepsize(t0, rm_info.stepsize_info);

    // for (int i = 0; i < Cache::N_DATA_ARRAYS; i++)
    // {
    //     float4 interpol_data = tex3D<float4>(data_array_textures.data_tex_linear[i], data_array_point);

    //     if (i == 0)
    //     {
    //         float density = interpol_data.x;
    //         float alpha = clamp(1.0f - exp(-density * dt), 0.0f, 1.0f);
    //         interpol_data.x = log(density) * weight;
    //     }

    //     setNetworkIntermediates_perf(mlp_head_network_inputs, interpol_data / weight, sample_idx, batch_size, i);
    // }

    const float3 lerp_t = applyInterpolationFunction(froxel_point - 0.5f - froxel_idx, cache_settings.interpol_function);

    bool isset[2][2][2] = {{{false, false}, {false, false}}, {{false, false}, {false, false}}};
    float3 data_array_points[2][2][2];
    float weight_data[2][2][2] = {{{0.0f, 0.0f}, {0.0f, 0.0f}}, {{0.0f, 0.0f}, {0.0f, 0.0f}}};

    for (int x = 0; x < 2; x++)
        for (int y = 0; y < 2; y++)
            for (int z = 0; z < 2; z++)
            {
                float3 froxel_point_tmp = froxel_idx + make_float3(x, y, z);
                bool in_cache_fov = gridContainsXY(froxel_point_tmp, froxel_grid_dims);
                bool inside_grid = in_cache_fov && inRange(froxel_point_tmp.z, 0, froxel_grid_dims.z);

                if (inside_grid)
                {
                    const int3 froxel_brick_idx = make_int3(froxel_point_tmp / Cache::BRICK_SIZE);
                    const float3 intra_brick_point = froxel_point_tmp - make_float3(froxel_brick_idx * Cache::BRICK_SIZE);

                    int packed_brick_index3D = tex3D<int>(froxel_packed_brick_index_array_3D, froxel_brick_idx);
                    bool brick_isset = packed_brick_index3D >= 0;
                    if (brick_isset)
                    {
                        isset[x][y][z] = true;
                        float3 data_array_point = expand3D(unpack3D(packed_brick_index3D), Cache::BRICK_SIZE, intra_brick_point, Cache::BRICK_PADDING);

                        data_array_points[x][y][z] = data_array_point;
                        weight_data[x][y][z] = tex3D<float>(data_array_textures.data_isset_tex_pt, data_array_point);
                    }
                }
            }

    float weight = lerp(weight_data, lerp_t);

    const float t0 = sample_t0[sample_idx];
    const float dt = calculate_stepsize(t0, rm_info.stepsize_info);

    for (int i = 0; i < Cache::N_DATA_ARRAYS; i++)
    {
        float4 data[2][2][2] = {{{make_float4(0.0f), make_float4(0.0f)}, {make_float4(0.0f), make_float4(0.0f)}}, {{make_float4(0.0f), make_float4(0.0f)}, {make_float4(0.0f), make_float4(0.0f)}}};
        for (int x = 0; x < 2; x++)
            for (int y = 0; y < 2; y++)
                for (int z = 0; z < 2; z++)
                {
                    if (isset[x][y][z])
                    {
                        data[x][y][z] = tex3D<float4>(data_array_textures.data_tex_pt[i], data_array_points[x][y][z]);
                    }
                }
        float4 interpol_data = lerp(data, lerp_t);

        if (i == 0)
        {
            float density = interpol_data.x;
            float alpha = clamp(1.0f - exp(-density * dt), 0.0f, 1.0f);
            interpol_data.x = log(density) * weight;
        }

        setNetworkIntermediates_perf(mlp_head_network_inputs, interpol_data / weight, sample_idx, batch_size, i);
    }
}

float inline __device__ shfl_lerp_3D(const float val, const float3 lerp_t)
{
    const float z1 = val;
    const float z2 = __shfl_down_sync(0xffffffff, z1, 1);

    const float y1 = d_lerp(z1, z2, lerp_t.z);
    const float y2 = __shfl_down_sync(0xffffffff, y1, 2);

    const float x1 = d_lerp(y1, y2, lerp_t.y);
    const float x2 = __shfl_down_sync(0xffffffff, x1, 4);

    return d_lerp(x1, x2, lerp_t.x);
}

float4 inline __device__ shfl_lerp_3D(const float4 val, const float3 lerp_t)
{
    const float4 z1 = val;
    const float4 z2 = make_float4(__shfl_down_sync(0xffffffff, z1.x, 1), __shfl_down_sync(0xffffffff, z1.y, 1), __shfl_down_sync(0xffffffff, z1.z, 1), __shfl_down_sync(0xffffffff, z1.w, 1));

    const float4 y1 = lerp(z1, z2, lerp_t.z);
    const float4 y2 = make_float4(__shfl_down_sync(0xffffffff, y1.x, 2), __shfl_down_sync(0xffffffff, y1.y, 2), __shfl_down_sync(0xffffffff, y1.z, 2), __shfl_down_sync(0xffffffff, y1.w, 2));

    const float4 x1 = lerp(y1, y2, lerp_t.y);
    const float4 x2 = make_float4(__shfl_down_sync(0xffffffff, x1.x, 4), __shfl_down_sync(0xffffffff, x1.y, 4), __shfl_down_sync(0xffffffff, x1.z, 4), __shfl_down_sync(0xffffffff, x1.w, 4));

    return lerp(x1, x2, lerp_t.x);
}

__global__ void generateNetworkInput_cache_shfl_perf_kernel(int total_n_threads,
                                                            int batch_size,

                                                            const RaymarchInfo rm_info,
                                                            const CacheSettings cache_settings,

                                                            const cudaTextureObject_t known_ranges_to,
                                                            const char *froxel_brick_isset_array,
                                                            const int *froxel_brick_index_array,
                                                            cudaTextureObject_t froxel_packed_brick_index_array_3D,
                                                            const int3 froxel_grid_bricks_per_dims,
                                                            const int3 froxel_grid_dims,

                                                            const int3 data_array_bricks_per_dim,
                                                            Cache::Textures data_array_textures,

                                                            const float *__restrict__ sample_t0,
                                                            const float3 *__restrict__ sample_pos,
                                                            float *__restrict__ sample_density,
                                                            half *__restrict__ mlp_head_network_inputs)
{
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= total_n_threads)
        return;

    const int sample_idx = idx / 8;

    const int lane_id = idx - sample_idx * 8;
    const int x = (lane_id & (1 << 2)) >> 2;
    const int y = (lane_id & (1 << 1)) >> 1;
    const int z = (lane_id & (1 << 0)) >> 0;

    const float3 froxel_point = sample_pos[sample_idx];
    float3 froxel_idx = floorf(froxel_point - 0.5f);

    // static constexpr CacheSettings cache_settings = CachePerformanceRenderer::default_cache_settings;
    const float3 lerp_t = applyInterpolationFunction(froxel_point - 0.5f - froxel_idx, cache_settings.interpol_function);

    float3 froxel_point_tmp = froxel_idx + make_float3(x, y, z);
    bool in_cache_fov = gridContainsXY(froxel_point_tmp, froxel_grid_dims);
    bool inside_grid = in_cache_fov && inRange(froxel_point_tmp.z, 0, froxel_grid_dims.z);

    const float t0 = sample_t0[sample_idx];
    const float dt = calculate_stepsize(t0, rm_info.stepsize_info);

    float weight_data = 0.0f;
    float data[4 * Cache::N_DATA_ARRAYS] = {0.0f};

    if (inside_grid)
    {
        const int3 froxel_brick_idx = make_int3(froxel_point_tmp / Cache::BRICK_SIZE);
        const float3 intra_brick_point = froxel_point_tmp - make_float3(froxel_brick_idx * Cache::BRICK_SIZE);

        int packed_brick_index3D = tex3D<int>(froxel_packed_brick_index_array_3D, froxel_brick_idx);
        bool brick_isset = packed_brick_index3D >= 0;
        if (brick_isset)
        {
            float3 data_array_point = expand3D(unpack3D(packed_brick_index3D), Cache::BRICK_SIZE, intra_brick_point, Cache::BRICK_PADDING);

            weight_data = tex3D<float>(data_array_textures.data_isset_tex_pt, data_array_point);
            for (int i = 0; i < Cache::N_DATA_ARRAYS; i++)
            {
                float4 tmp = tex3D<float4>(data_array_textures.data_tex_pt[i], data_array_point);
                data[i * 4 + 0] = tmp.x;
                data[i * 4 + 1] = tmp.y;
                data[i * 4 + 2] = tmp.z;
                data[i * 4 + 3] = tmp.w;
            }
        }
    }

    float weight_interpol = shfl_lerp_3D(weight_data, lerp_t);

    for (int i = 0; i < 4 * Cache::N_DATA_ARRAYS; i++)
    {
        float data_interpol = shfl_lerp_3D(data[i], lerp_t);

        if (i == 0)
        {
            float density = data_interpol;
            float alpha = clamp(1.0f - exp(-density * dt), 0.0f, 1.0f);
            data_interpol = log(density) * weight_interpol;
        }

        if (lane_id == 0)
            mlp_head_network_inputs[sample_idx + i * batch_size] = __float2half(data_interpol / weight_interpol);
    }
}

float inline __device__ shfl_reduce_sum_8(float val)
{
    val += __shfl_down_sync(0xffffffff, val, 1);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 4);
    return val;
}

template <bool FIRST_LATENT_IS_DENSITY = false>
__global__ void generateNetworkInput_cache_shfl_perf_kernel_cg(int total_n_threads,
                                                               int batch_size,

                                                               const RaymarchInfo rm_info,
                                                               const CacheSettings cache_settings,

                                                               const cudaTextureObject_t known_ranges_to,
                                                               const char *froxel_brick_isset_array,
                                                               const int *froxel_brick_index_array,
                                                               cudaTextureObject_t froxel_packed_brick_index_array_3D,
                                                               const int3 froxel_grid_bricks_per_dims,
                                                               const int3 froxel_grid_dims,

                                                               const int3 data_array_bricks_per_dim,
                                                               Cache::Textures data_array_textures,

                                                               const float *__restrict__ sample_t0,
                                                               const float3 *__restrict__ sample_pos,
                                                               half *__restrict__ mlp_head_network_inputs)
{
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= total_n_threads)
        return;

    const int sample_idx = idx / 8;

    const int lane_id = idx - sample_idx * 8;
    const int x = (lane_id & (1 << 2)) >> 2;
    const int y = (lane_id & (1 << 1)) >> 1;
    const int z = (lane_id & (1 << 0)) >> 0;

    const float3 froxel_point = sample_pos[sample_idx];
    float3 froxel_idx = floorf(froxel_point - 0.5f);

    // static constexpr CacheSettings cache_settings = CachePerformanceRenderer::default_cache_settings;
    const float3 lerp_t = applyInterpolationFunction(froxel_point - 0.5f - froxel_idx, cache_settings.interpol_function);

    const float3 xyz_offset = make_float3(x, y, z);
    const float3 froxel_point_tmp = froxel_idx + xyz_offset;
    const float3 interpol_weights = (1.0f - xyz_offset) * (1.0f - 2.0f * lerp_t) + lerp_t;

    float interpol_weight = interpol_weights.x * interpol_weights.y * interpol_weights.z;
    bool in_cache_fov = gridContainsXY(froxel_point_tmp, froxel_grid_dims);
    bool inside_grid = in_cache_fov && inRange(froxel_point_tmp.z, 0, froxel_grid_dims.z);

    const float t0 = sample_t0[sample_idx];
    const float dt = calculate_stepsize(t0, rm_info.stepsize_info);

    float weight_data = 0.0f;
    float density_data = 0.0f;

    bool brick_isset = false;
    float3 data_array_point;

    if (inside_grid)
    {
        const int3 froxel_brick_idx = make_int3(froxel_point_tmp / Cache::BRICK_SIZE);
        const float3 intra_brick_point = froxel_point_tmp - make_float3(froxel_brick_idx * Cache::BRICK_SIZE);

        int packed_brick_index3D = tex3D<int>(froxel_packed_brick_index_array_3D, froxel_brick_idx);
        brick_isset = packed_brick_index3D >= 0;
        if (brick_isset)
        {
            data_array_point = expand3D(unpack3D(packed_brick_index3D), Cache::BRICK_SIZE, intra_brick_point, Cache::BRICK_PADDING);
            weight_data = interpol_weight * tex3D<float>(data_array_textures.data_isset_tex_pt, data_array_point);

            if (!FIRST_LATENT_IS_DENSITY)
                density_data = interpol_weight * tex3D<float>(data_array_textures.data_density_alpha_tex_pt, data_array_point);
        }
    }

    float weight_interpol = shfl_reduce_sum_8(weight_data);

    if (!FIRST_LATENT_IS_DENSITY)
    {
        float density_interpol = shfl_reduce_sum_8(density_data);

        if (lane_id == 0)
            mlp_head_network_inputs[sample_idx] = __float2half(log(max(density_interpol, 1e-6f)));
    }

    for (int i = 0; i < Cache::N_DATA_ARRAYS; i++)
    {
        float data[4] = {0.0f};
        if (brick_isset)
        {
            float4 tmp = interpol_weight * tex3D<float4>(data_array_textures.data_tex_pt[i], data_array_point);
            data[0] = tmp.x;
            data[1] = tmp.y;
            data[2] = tmp.z;
            data[3] = tmp.w;
        }

        for (int j = 0; j < 4; j++)
        {
            float data_interpol = shfl_reduce_sum_8(data[j]);

            if (i == 0 && j == 0 && FIRST_LATENT_IS_DENSITY)
            {
                float density = data_interpol;
                float alpha = clamp(1.0f - exp(-density * dt), 0.0f, 1.0f);
                data_interpol = log(density) * weight_interpol;
            }

            if (lane_id == 0)
                mlp_head_network_inputs[sample_idx + (i * 4 + j + (FIRST_LATENT_IS_DENSITY ? 0 : 1)) * batch_size] = __float2half(data_interpol / weight_interpol);
        }
    }
}

template <bool DEBUG>
__global__ void accumulate_cache_perf_kernel(int n_alive,
                                             int batch_size_resample,
                                             int batch_size_cache,
                                             const RaymarchInfo rm_info,
                                             const SceneInfo scene_info,
                                             BaseRayResult *__restrict__ results,
                                             const BaseRayInitInfo *__restrict__ ray_inits,
                                             const CachePerformanceRenderer::SubiterRayEntryCumsum *__restrict__ subiter_ray_entries_cumsum,
                                             int *__restrict__ alive_buffer,
                                             const half *__restrict__ network_output_resample_rgbs,
                                             const half *__restrict__ network_output_cache_rgbs,
                                             const float *__restrict__ sample_resample_t0,
                                             const float *__restrict__ sample_cache_t0,
                                             cudaSurfaceObject_t buffer_contrib_resample,
                                             cudaSurfaceObject_t buffer_contrib_cache)
{
    const int curr_subiter_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (curr_subiter_idx >= n_alive)
        return;

    BaseRayResult result = results[curr_subiter_idx];
    bool alive = alive_buffer[curr_subiter_idx];

    const auto ray_entry_cumsum = subiter_ray_entries_cumsum[curr_subiter_idx];
    const auto ray_entry_cumsum_prev = curr_subiter_idx > 0 ? subiter_ray_entries_cumsum[curr_subiter_idx - 1] : CachePerformanceRenderer::SubiterRayEntryCumsum{0, 0, 0, 0};
    const auto ray_entry = CachePerformanceRenderer::SubiterRayEntry::createFromDiff(ray_entry_cumsum, ray_entry_cumsum_prev);

    int n_steps_resample = ray_entry.n_samples_resample;
    int n_steps_cache = ray_entry.n_samples_cache;

    int offset_resample = ray_entry_cumsum.sample_offset_resample - n_steps_resample;
    int offset_cache = ray_entry_cumsum.sample_offset_cache - n_steps_cache;

    int step_resample = 0;
    int step_cache = 0;

    float accum_weight_resample = 0.0f;
    float accum_weight_cache = 0.0f;

    while (step_resample < n_steps_resample || step_cache < n_steps_cache)
    {
        int row_offset_resample = offset_resample + step_resample;
        int row_offset_cache = offset_cache + step_cache;

        float t0;
        const float t0_resample = step_resample < n_steps_resample ? sample_resample_t0[row_offset_resample] : 1e10f;
        const float t0_cache = step_cache < n_steps_cache ? sample_cache_t0[row_offset_cache] : 1e10f;

        bool is_cache_sample;

        float3 tmp_network_output_rgb;
        float tmp_network_output_density;
        if (step_cache >= n_steps_cache || t0_resample < t0_cache)
        {
            tmp_network_output_rgb.x = __half2float(network_output_resample_rgbs[row_offset_resample + 0 * batch_size_resample]);
            tmp_network_output_rgb.y = __half2float(network_output_resample_rgbs[row_offset_resample + 1 * batch_size_resample]);
            tmp_network_output_rgb.z = __half2float(network_output_resample_rgbs[row_offset_resample + 2 * batch_size_resample]);
            tmp_network_output_density = __half2float(network_output_resample_rgbs[row_offset_resample + 3 * batch_size_resample]);
            t0 = t0_resample;

            step_resample++;
            is_cache_sample = false;
        }
        else
        {
            tmp_network_output_rgb.x = __half2float(network_output_cache_rgbs[row_offset_cache + 0 * batch_size_cache]);
            tmp_network_output_rgb.y = __half2float(network_output_cache_rgbs[row_offset_cache + 1 * batch_size_cache]);
            tmp_network_output_rgb.z = __half2float(network_output_cache_rgbs[row_offset_cache + 2 * batch_size_cache]);
            tmp_network_output_density = __half2float(network_output_cache_rgbs[row_offset_cache + 3 * batch_size_cache]);
            t0 = t0_cache;

            step_cache++;
            is_cache_sample = true;
        }

        const float3 rgb = sigmoid(tmp_network_output_rgb);
        const float density = exp(tmp_network_output_density);

        const float dt = calculate_stepsize(t0, rm_info.stepsize_info);

        float alpha = clamp(1.0f - exp(-density * dt), 0.0f, 1.0f);

        if (alpha > scene_info.alpha_thre)
        {
            float weight = alpha * result.transmittance;
            result.rgb += weight * rgb;
            result.transmittance *= (1.0f - alpha);

            if (!is_cache_sample)
                accum_weight_resample += weight;
            else
                accum_weight_cache += weight;
        }

        if (result.transmittance < 1e-4f)
        {
            alive = false;
            break;
        }
    }

    if constexpr (DEBUG)
    {
        int ray_id = ray_inits[curr_subiter_idx].ray_id;

        int y = ray_id / rm_info.cam_info.resolution.x;
        int x = ray_id % rm_info.cam_info.resolution.x;

        float prev;
        surf2Dread(&prev, buffer_contrib_resample, x * sizeof(float), y);
        surf2Dwrite(prev + accum_weight_resample, buffer_contrib_resample, x * sizeof(float), y);

        surf2Dread(&prev, buffer_contrib_cache, x * sizeof(float), y);
        surf2Dwrite(prev + accum_weight_cache, buffer_contrib_cache, x * sizeof(float), y);
    }

    alive_buffer[curr_subiter_idx] = alive;
    results[curr_subiter_idx] = result;
}

__global__ void writeImageBuffer_cache_perf_kernel(const int2 resolution,
                                                   const BaseRayResult *__restrict__ final_results,
                                                   cudaSurfaceObject_t image_buffer, const int sample_index)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= resolution.x || y >= resolution.y)
        return;

    const int ray_id = y * resolution.x + x;

    BaseRayResult result = final_results[ray_id];
    // result.rgb = make_float3(clamp((result.total_n_steps_resample) / 128.f, 0.0f, 1.0f));
    // result.transmittance = 0.0f;

    float3 rgb = result.rgb;
    float transmittance = result.transmittance;

    rgb = applyWhiteBackground(rgb, transmittance);
    uchar4 rgb_8_prev;
    surf2Dread(&rgb_8_prev, image_buffer, x * sizeof(uchar4), y);
    uchar4 rgb_8_new = color_to_uchar4((rgb + uchar4_to_color(rgb_8_prev) * sample_index) / float(sample_index + 1));

    surf2Dwrite(rgb_8_new, image_buffer, x * sizeof(uchar4), y);
}

__global__ void writeImageBuffer_cache_perf_postRender_kernel(const int2 resolution,
                                                              const BaseRayResult *__restrict__ final_results,
                                                              cudaSurfaceObject_t image_buffer)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= resolution.x || y >= resolution.y)
        return;

    const int ray_id = y * resolution.x + x;

    BaseRayResult result = final_results[ray_id];

    float3 rgb = make_float3(1.0f);
    surf2Dwrite(color_to_uchar4(rgb), image_buffer, x * sizeof(uchar4), y);
}

void writeImageBuffer_cache_perf(const int2 resolution, const BaseRayResult *final_results, RenderBuffer &image_buffer, const int sample_index)
{
    constexpr int BLOCK_SIZE_2D = 16;
    dim3 block_size_2D(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
    dim3 grid_size_2D(toDim3(divRoundUp(resolution, BLOCK_SIZE_2D)));
    writeImageBuffer_cache_perf_kernel<<<grid_size_2D, block_size_2D>>>(resolution, final_results, image_buffer.surface(), sample_index);
}

CachePerformanceRenderer::CachePerformanceRenderer(SceneInfo scene_info, CudaBuffer<uint8_t> &occupancy_grid, NerfNetwork<half> &nerf_network, Cache *cache, Cache *next_cache)
    : CacheRenderer(scene_info, occupancy_grid, nerf_network, cache, next_cache)
{
    if (nerf_network.latent_width() > Cache::N_DATA_ARRAYS * 4)
        throw std::runtime_error("Latent width of the model is larger than the (hard-coded) cache size; Change N_DATA_ARRAYS!");

#ifdef RTX_ENABLED
    if (INIT_RAYS_WITH_RTX || SAMPLE_WITH_RTX)
    {
        std::cout << "Initializing OWL..." << std::endl;
        _context = owlContextCreate(nullptr, 1);
        OWLGeomType triangle_geom_type = createGeomType(_context);

        createMeshFromGridCompact(_occupancy_grid_mesh, scene_info, occupancy_grid, _context);
        _grid_geom = buildAccel(_occupancy_grid_mesh, triangle_geom_type, _context);

        if (INIT_RAYS_WITH_RTX)
        {
            _init_rays_module = owlModuleCreate(_context, init_ray_payloads_ptx);
            _init_rays_program.init(_context, _init_rays_module, triangle_geom_type);
        }

        if (SAMPLE_WITH_RTX)
        {
            _sample_module = owlModuleCreate(_context, sample_segments_ptx);
            _sample_program.init(_context, _sample_module, triangle_geom_type);
        }

        owlBuildPrograms(_context);
        owlBuildPipeline(_context);

        // to not have initial startup costs in render function
        if (INIT_RAYS_WITH_RTX)
            _init_rays_program.dummyLaunch(_context);
        if (SAMPLE_WITH_RTX)
            _sample_program.dummyLaunch(_context);
    }
#endif
}

void CachePerformanceRenderer::resizeRenderbuffers(int2 resolution)
{
    int n_rays = resolution.x * resolution.y;
    int buffer_size = tcnn::next_multiple(max(n_rays, TARGET_BATCH_SIZE) * MIN_N_STEPS_PER_SUBITER, (int)tcnn::batch_size_granularity);

    if (n_rays != _ray_order_buffer.size())
    {
        _ray_order_buffer.resize(n_rays);
        createRayOrder(resolution, _ray_order_buffer);
    }

    if (_force_renderbuffer_resize || n_rays > _payloads_final.size())
    {
        // if caching resolution and render resolution are different, keep size to the larger one
        _alive_counter.resize(1);
        _alive_buffer.resize(n_rays);
        _new_index_buffer.resize(n_rays);

        size_t temp_storage_bytes;
        CUDA_CHECK_THROW(cub::DeviceScan::InclusiveSum(nullptr, temp_storage_bytes, _alive_buffer.data(), _new_index_buffer.data(), n_rays));
        constexpr size_t temp_storage_type_size = sizeof(decltype(_cub_alive_buffer_tmp)::type);
        _cub_alive_buffer_tmp.resize(round_up_pow2(temp_storage_bytes, temp_storage_type_size) / temp_storage_type_size);

        _subiter_compact_indices_resample.resize(n_rays);
        _subiter_compact_indices_cache.resize(n_rays);

        _subiter_ray_entries_cumsum.resize(n_rays);

        CUDA_CHECK_THROW(cub::DeviceScan::InclusiveSum(nullptr, temp_storage_bytes, (int4 *)_subiter_ray_entries_cumsum.data(), (int4 *)_subiter_ray_entries_cumsum.data(), n_rays));
        constexpr size_t temp_storage_int4_type_size = sizeof(decltype(_cub_alive_int4_buffer_tmp)::type);
        _cub_alive_int4_buffer_tmp.resize(round_up_pow2(temp_storage_bytes, temp_storage_int4_type_size) / temp_storage_int4_type_size);

        _payloads_doublebuffer.resize(n_rays * 2);
        _payloads_final.resize(n_rays);

        _ray_dblbuffer.resize(n_rays);
        _ray_results_final.resize(n_rays);

        _samples_resample_t0_tmp.resize(buffer_size);
        _samples_resample.resize(buffer_size);

        _samples_cache_t0_tmp.resize(buffer_size);
        _samples_cache_tmp.resize(buffer_size);
        _samples_cache_init_viewdir.resize(buffer_size);
        _samples_cache_froxel_pos.resize(buffer_size);
        _samples_cache.resize(buffer_size);

        _network_intermediate_buffer.resize(buffer_size * _nerf_network.required_buffer_width());
        _network_output_resample_rgbs.resize(buffer_size * _nerf_network.padded_output_width());
        _network_output_cache_rgbs.resize(buffer_size * _nerf_network.padded_output_width());

        _force_renderbuffer_resize = false;
        CUDA_SYNC_CHECK_THROW();
    }
}

void CachePerformanceRenderer::render(RaymarchInfo rm_info, RenderBuffer &image_buffer, DebugData *debug_data)
{
    std::scoped_lock lk(_read_mutex);

    if (!_cache->_initialized)
        throw std::runtime_error("Cache is not initialized!");

    resizeRenderbuffers(rm_info.cam_info.resolution);

    int n_rays = rm_info.cam_info.resolution.x * rm_info.cam_info.resolution.y;
    int n_alive = n_rays;

    _ray_dblbuffer.reset();

    _timer.start(TimingState::InitRays);
#ifdef RTX_ENABLED
    if (INIT_RAYS_WITH_RTX)
    {
        _init_rays_program.launch(_context, _grid_geom, rm_info, _scene_info, _alive_buffer.data(),
                                  _ray_dblbuffer.init_infos.data(), _ray_dblbuffer.payloads.data(), _ray_dblbuffer.results.data(),
                                  _ray_order_buffer.data());
    }
    else
#endif
    {
        auto init_buffer = _ray_dblbuffer.getPrevBuffer();
        initRayPayloads_split(rm_info, _scene_info, init_buffer, _alive_buffer.data(), _ray_order_buffer.data());
    }

    int n_cache_samples_evaluated_total = 0;
    int n_resample_samples_evaluated_total = 0;
    int subiter_n_resample_total = 0;
    int subiter_n_cache_total = 0;

    int n_steps_curr_subiter = clamp(TARGET_BATCH_SIZE / n_alive, MIN_N_STEPS_PER_SUBITER, MAX_N_STEPS_PER_SUBITER);
    bool first_latent_is_density = cache_settings.interpol_variant == DensityInterpolVariant::DensityIntermediates;

    int subiter = 0;
    while (subiter < MAX_SUBITER)
    {
        auto prev_rays = _ray_dblbuffer.getPrevBuffer();
        auto curr_rays = _ray_dblbuffer.getCurrBuffer();

#ifdef RTX_ENABLED
        if (SAMPLE_WITH_RTX)
        {
            _timer.start(TimingState::SampleRtx);
            _sample_program.launch(_context, _grid_geom, n_alive, n_steps_curr_subiter, rm_info, _scene_info, prev_rays.init_infos, prev_rays.payloads, _alive_buffer.data());
        }
#endif

        size_t tmp_size_bytes = _cub_alive_buffer_tmp.sizeInBytes();
        _timer.start(TimingState::CompactRays);
        CUDA_CHECK_THROW(cub::DeviceScan::InclusiveSum(_cub_alive_buffer_tmp.data(), tmp_size_bytes, _alive_buffer.data(), _new_index_buffer.data(), n_alive));
        n_alive = compactPayloads_coherent_split(n_alive, _new_index_buffer.data(), _alive_buffer.data(), prev_rays, curr_rays, _ray_results_final.data());
        if (n_alive == 0)
            break;

        n_steps_curr_subiter = clamp(TARGET_BATCH_SIZE / n_alive, MIN_N_STEPS_PER_SUBITER, MAX_N_STEPS_PER_SUBITER);

        _timer.syncElapsed();
        _timer.start(TimingState::Sample);

#define SAMPLE_CACHE_PERFORMANCE_KERNEL_CALL(FIRST_LATENT_IS_DENSITY)                                                                                                                                                    \
    tcnn::linear_kernel(sample_cache_performance_kernel<FIRST_LATENT_IS_DENSITY>, 0, cudaStreamDefault,                                                                                                                  \
                        n_alive, n_steps_curr_subiter, rm_info, _cache->_rm_info, _scene_info, _render_mode, _occupancy_grid.data(),                                                                                     \
                        curr_rays.init_infos, curr_rays.payloads, _alive_buffer.data(), _subiter_ray_entries_cumsum.data(),                                                                                              \
                        _cache->_known_ranges_to.texturePt(), _cache->_brick_isset_array, _cache->_brick_index_array, _cache->_packed_brick_index_array_3D.texturePt(), _cache->brickArrayDims(), _cache->frustumDims(), \
                        _cache->_data_isset_array.texturePt(), _cache->textures(), _cache->dataArrayBricksPerDim(),                                                                                                      \
                        _samples_cache_t0_tmp.data(), _samples_cache_tmp.data(), _samples_resample_t0_tmp.data())

        if (first_latent_is_density)
            SAMPLE_CACHE_PERFORMANCE_KERNEL_CALL(true);
        else
            SAMPLE_CACHE_PERFORMANCE_KERNEL_CALL(false);
#undef SAMPLE_CACHE_PERFORMANCE_KERNEL_CALL

        int4 tmp_cumsum;
        tmp_size_bytes = _cub_alive_int4_buffer_tmp.sizeInBytes();
        CUDA_CHECK_THROW(cub::DeviceScan::InclusiveSum(_cub_alive_int4_buffer_tmp.data(), tmp_size_bytes, (int4 *)_subiter_ray_entries_cumsum.data(), (int4 *)_subiter_ray_entries_cumsum.data(), n_alive));
        CUDA_CHECK_THROW(cudaMemcpy(&tmp_cumsum, (int4 *)_subiter_ray_entries_cumsum.data() + (n_alive - 1), sizeof(int4), cudaMemcpyDeviceToHost));
        SubiterEntryCumsumTotal subiter_entries_total{tmp_cumsum.x, tmp_cumsum.y, tmp_cumsum.z, tmp_cumsum.w};

        _timer.start(TimingState::PrepareSamplesResample);
        tcnn::linear_kernel(prepareNetworkGeneration_perf_kernel, 0, cudaStreamDefault,
                            n_alive, _subiter_ray_entries_cumsum.data(), _subiter_compact_indices_resample.data(), _subiter_compact_indices_cache.data());

        // std::cout << "Rays: " << subiter_entries_total.n_rays_resample << " + " << subiter_entries_total.n_rays_cache << ", " << n_steps_curr_subiter << std::endl;
        // std::cout << "Samples: " << subiter_entries_total.n_samples_resample << " + " << subiter_entries_total.n_samples_cache << ", " << n_steps_curr_subiter << std::endl;

        int curr_batch_size_resample = tcnn::next_multiple(subiter_entries_total.n_samples_resample, (int)tcnn::batch_size_granularity);
        tcnn::linear_kernel(generateNetworkInput_resample_perf_kernel, 0, cudaStreamDefault,
                            subiter_entries_total.n_rays_resample * n_steps_curr_subiter, n_alive, subiter_entries_total.n_samples_resample, n_steps_curr_subiter, rm_info, _scene_info, curr_rays.init_infos,
                            _subiter_ray_entries_cumsum.data(), _subiter_compact_indices_resample.data(),
                            _samples_resample_t0_tmp.data(), _samples_resample.t0.data(), _samples_resample.pos.data(), _samples_resample.dir.data());

        _timer.start(TimingState::InferenceResample);
        if (curr_batch_size_resample > 0)
        {
            tcnn::GPUMatrix<float> network_input_pos((float *)_samples_resample.pos.data(), 3, curr_batch_size_resample);
            tcnn::GPUMatrix<float> network_input_dir((float *)_samples_resample.dir.data(), 3, curr_batch_size_resample);

            tcnn::GPUMatrix<half, tcnn::RM> mlp_head_input_matrix(_network_intermediate_buffer.data(), _nerf_network.m_mlp_head->input_width(), curr_batch_size_resample);
            tcnn::GPUMatrix<half, tcnn::RM> rgbsigma_matrix(_network_output_resample_rgbs.data(), _nerf_network.padded_output_width(), curr_batch_size_resample);

            _nerf_network.inference(cudaStreamDefault, network_input_pos, network_input_dir, network_input_dir, mlp_head_input_matrix, rgbsigma_matrix);
            n_resample_samples_evaluated_total += curr_batch_size_resample;
        }

        int curr_batch_size_cache = tcnn::next_multiple(subiter_entries_total.n_samples_cache, (int)tcnn::batch_size_granularity);

        tcnn::GPUMatrix<float, tcnn::RM> network_input_pos_cache((float *)_samples_cache.pos.data(), 3, curr_batch_size_cache);
        tcnn::GPUMatrix<float, tcnn::RM> network_input_dir_cache((float *)_samples_cache.dir.data(), 3, curr_batch_size_cache);
        tcnn::GPUMatrix<float, tcnn::RM> network_input_init_viewdir_cache((float *)_samples_cache_init_viewdir.data(), 3, curr_batch_size_cache);

        _timer.start(TimingState::PrepareSamplesCache);

#define GENERATE_NETWORK_INPUT_CACHE_CALL(FIRST_LATENT_IS_DENSITY)                                                                                                                                                           \
    if (Cache::BRICK_PADDING > 0 && cache_settings.interpol_function == InterpolFunction::Linear)                                                                                                                            \
    {                                                                                                                                                                                                                        \
        tcnn::linear_kernel(generateNetworkInput_cache_padded_kernel<FIRST_LATENT_IS_DENSITY>, 0, cudaStreamDefault,                                                                                                         \
                            subiter_entries_total.n_rays_cache *n_steps_curr_subiter, subiter_entries_total.n_rays_cache, n_alive, subiter_entries_total.n_samples_cache, n_steps_curr_subiter,                              \
                            rm_info, _cache->_rm_info, _scene_info, curr_rays.init_infos,                                                                                                                                    \
                            _subiter_ray_entries_cumsum.data(), _subiter_compact_indices_cache.data(),                                                                                                                       \
                            _samples_cache_tmp.data(), _samples_cache.t0.data(), _samples_cache_froxel_pos.data(), network_input_pos_cache.view(), network_input_dir_cache.view(), network_input_init_viewdir_cache.view(),  \
                            curr_batch_size_cache, _cache->_packed_brick_index_array_3D.texturePt(), _cache->brickArrayDims(), _cache->frustumDims(),                                                                        \
                            _cache->dataArrayBricksPerDim(), _cache->textures(),                                                                                                                                             \
                            _network_intermediate_buffer.data());                                                                                                                                                            \
    }                                                                                                                                                                                                                        \
    else                                                                                                                                                                                                                     \
    {                                                                                                                                                                                                                        \
        tcnn::linear_kernel(generateNetworkInput_preparation_perf_kernel<FIRST_LATENT_IS_DENSITY>, 0, cudaStreamDefault,                                                                                                     \
                            subiter_entries_total.n_rays_cache *n_steps_curr_subiter, subiter_entries_total.n_rays_cache, n_alive, subiter_entries_total.n_samples_cache, n_steps_curr_subiter,                              \
                            rm_info, _cache->_rm_info, _scene_info, curr_rays.init_infos,                                                                                                                                    \
                            _subiter_ray_entries_cumsum.data(), _subiter_compact_indices_cache.data(),                                                                                                                       \
                            _samples_cache_t0_tmp.data(), _samples_cache.t0.data(), _samples_cache_froxel_pos.data(),                                                                                                        \
                            network_input_pos_cache.view(), network_input_dir_cache.view(), network_input_init_viewdir_cache.view());                                                                                        \
                                                                                                                                                                                                                             \
        tcnn::linear_kernel(generateNetworkInput_cache_shfl_perf_kernel_cg<FIRST_LATENT_IS_DENSITY>, 0, cudaStreamDefault,                                                                                                   \
                            subiter_entries_total.n_samples_cache * 8, curr_batch_size_cache, rm_info, cache_settings,                                                                                                       \
                            _cache->_known_ranges_to.texturePt(), _cache->_brick_isset_array, _cache->_brick_index_array, _cache->_packed_brick_index_array_3D.texturePt(), _cache->brickArrayDims(), _cache->frustumDims(), \
                            _cache->dataArrayBricksPerDim(), _cache->textures(),                                                                                                                                             \
                            _samples_cache.t0.data(), _samples_cache_froxel_pos.data(), _network_intermediate_buffer.data());                                                                                                \
    }

        if (first_latent_is_density)
        {
            GENERATE_NETWORK_INPUT_CACHE_CALL(true);
        }
        else
        {
            GENERATE_NETWORK_INPUT_CACHE_CALL(false);
        };
#undef GENERATE_NETWORK_INPUT_CACHE_CALL

        subiter_n_resample_total += subiter_entries_total.n_samples_resample;
        subiter_n_cache_total += subiter_entries_total.n_samples_cache;

        // std::cout << "Resample: " << subiter_entries_total.n_samples_resample << "|" << ", Cache: " << subiter_entries_total.n_samples_cache << std::endl;

        _timer.start(TimingState::InferenceCache);
        if (curr_batch_size_cache > 0)
        {
            tcnn::GPUMatrix<half, tcnn::RM> mlp_head_input_matrix(_network_intermediate_buffer.data(), _nerf_network.m_mlp_head->input_width(), curr_batch_size_cache);
            tcnn::GPUMatrix<half, tcnn::RM> rgbsigma_matrix_cache(_network_output_cache_rgbs.data(), _nerf_network.padded_output_width(), curr_batch_size_cache);

            _nerf_network.inferenceHead(cudaStreamDefault, network_input_pos_cache, network_input_dir_cache, network_input_init_viewdir_cache, mlp_head_input_matrix, rgbsigma_matrix_cache);
            n_cache_samples_evaluated_total += curr_batch_size_cache;
        }

        _timer.start(TimingState::Accumulate);

#define ACCUMULATE_CACHE_PERF_CALL(DEBUG)                                                                                                        \
    tcnn::linear_kernel(accumulate_cache_perf_kernel<DEBUG>, 0, cudaStreamDefault,                                                               \
                        n_alive, curr_batch_size_resample, curr_batch_size_cache, rm_info, _scene_info, curr_rays.results, curr_rays.init_infos, \
                        _subiter_ray_entries_cumsum.data(),                                                                                      \
                        _alive_buffer.data(), _network_output_resample_rgbs.data(), _network_output_cache_rgbs.data(),                           \
                        _samples_resample.t0.data(), _samples_cache.t0.data(),                                                                   \
                        DEBUG ? debug_data->buffers.at(0)->surface() : 0ULL,                                                                     \
                        DEBUG ? debug_data->buffers.at(1)->surface() : 0ULL)

        if (debug_data == nullptr)
            ACCUMULATE_CACHE_PERF_CALL(false);
        else
            ACCUMULATE_CACHE_PERF_CALL(true);
#undef ACCUMULATE_CACHE_PERF_CALL

        _ray_dblbuffer.advance();
        subiter++;
    }

    _render_stats.samples_ppx = (n_resample_samples_evaluated_total + n_cache_samples_evaluated_total) / (float)n_rays;
    _render_stats.cache_samples_evaluated = n_cache_samples_evaluated_total / (float)n_rays;
    _render_stats.resample_samples_evaluated = n_resample_samples_evaluated_total / (float)n_rays;
    _render_stats.n_resamples = n_resample_samples_evaluated_total;
    _render_stats.n_cache_hits = n_cache_samples_evaluated_total;

    _timer.start(TimingState::WriteImage);
    writeImageBuffer_cache_perf(rm_info.cam_info.resolution, _ray_results_final.data(), image_buffer, rm_info.sample_index);

    _timer.syncElapsed();

    // std::cout << "Cache: " << (n_resample_samples_evaluated_total + n_cache_samples_evaluated_total) << " = " << n_resample_samples_evaluated_total << " + " << n_cache_samples_evaluated_total << std::endl;
    // std::cout << ", Resample: " << subiter_n_resample_total / (float) n_resample_samples_evaluated_total << ", Cache: " << subiter_n_cache_total / (float) n_cache_samples_evaluated_total << std::endl;
}

void CachePerformanceRenderer::postRender(RaymarchInfo rm_info, RenderBuffer &additional_image_buffer)
{
    if (!_cache->_initialized)
        throw std::runtime_error("Cache is not initialized!");

    const int BLOCK_SIZE_2D = 16;
    dim3 block_size_2D(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
    dim3 image_grid_size_2D(toDim3(divRoundUp(rm_info.cam_info.resolution, BLOCK_SIZE_2D)));
    writeImageBuffer_cache_perf_postRender_kernel<<<image_grid_size_2D, block_size_2D>>>(rm_info.cam_info.resolution, _ray_results_final.data(), additional_image_buffer.surface());
}