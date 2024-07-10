/*
 * Copyright (C) 2024, Michael Steiner, Graz University of Technology.
 * This code is licensed under the MIT license.
 */

#include "cache_renderer.h"

#include "raymarch_common.h"
#include "util/cub_helper.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>

using namespace cooperative_groups;
namespace cg = cooperative_groups;

template<int BRICK_SIZE, int BRICK_PADDING>
__global__ void initPayloads_cache_init_kernel(const RaymarchInfo rm_info,
                                               const SceneInfo scene_info,
                                               const int3 frustum_dims,
                                               const int3 brick_array_dims,
                                               
                                               BlockPayloadCacheInit* __restrict__ block_payloads,
                                               RayPayloadCacheInit* __restrict__ ray_payloads,
                                               int* __restrict__ blocks_alive_buffer)
{
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<Cache::PADDED_BRICK_SIZE*Cache::PADDED_BRICK_SIZE> tile = cg::tiled_partition<Cache::PADDED_BRICK_SIZE*Cache::PADDED_BRICK_SIZE>(block);

    constexpr int PADDED_CACHE_BLOCK_SIZE = BRICK_SIZE + 2 * BRICK_PADDING;
    if (threadIdx.x >= PADDED_CACHE_BLOCK_SIZE || threadIdx.y >= PADDED_CACHE_BLOCK_SIZE)
        return;

    const int block_idx1D = blockIdx.y * brick_array_dims.x + blockIdx.x;
    const int2 block_offset = make_int2(blockIdx.x, blockIdx.y) * BRICK_SIZE;

    const int2 ray_idx_in_block = make_int2(threadIdx.x, threadIdx.y);
    const int ray_id_in_block1D = ray_idx_in_block.y * PADDED_CACHE_BLOCK_SIZE + ray_idx_in_block.x;
    const int ray_id_global1D = block_idx1D * PADDED_CACHE_BLOCK_SIZE * PADDED_CACHE_BLOCK_SIZE + ray_id_in_block1D;

    const int2 pixel = block_offset + ray_idx_in_block - BRICK_PADDING;
    const bool is_padding = ray_idx_in_block.x < BRICK_PADDING || ray_idx_in_block.x >= (BRICK_SIZE + BRICK_PADDING) || 
                            ray_idx_in_block.y < BRICK_PADDING || ray_idx_in_block.y >= (BRICK_SIZE + BRICK_PADDING);


    float2 pixel_offset = {0.5f, 0.5f};

    const float2 screen_point = make_float2(pixel) + pixel_offset;
    const FrustumRay ray = generateRay(ray_id_global1D, screen_point, rm_info);

    float aabb_t0, aabb_t1;
    int hits_aabb = ray_aabb_intersect(ray, scene_info.aabb_from, scene_info.aabb_to, aabb_t0, aabb_t1);
    int alive_interior = (!is_padding) * hits_aabb;
    int n_alive_interior_rays = cg::reduce(tile, alive_interior, cg::plus<int>());

    int start_step = max(0, (int) (rm_info.stepsize_info.step_from_t(hits_aabb ? aabb_t0 : rm_info.stepsize_info.near) - rm_info.stepsize_info.near_i));
    int group_min_interior_start_block_step = cg::reduce(tile, start_step + is_padding * 99999, cg::less<int>()) / BRICK_SIZE;

    BlockPayloadCacheInit block_payload{
        curr_block_step: group_min_interior_start_block_step,
        n_alive_interior_rays: n_alive_interior_rays,
        n_total_bricks: 0
    };
    block_payloads[block_idx1D] = block_payload;
    blocks_alive_buffer[block_idx1D] = n_alive_interior_rays > 0;

    RayPayloadCacheInit ray_payload{
        origin: ray.origin,
        dir: ray.dir,
        far: hits_aabb ? min(aabb_t1, rm_info.stepsize_info.far) : rm_info.stepsize_info.far,

        subiter_n_steps: 0,
        subiter_offset_in_block: 0,
        total_n_steps: 0,

        alive: hits_aabb,
        transmittance: 1.0f
    };
    ray_payloads[ray_id_global1D] = ray_payload;
}

template<int BRICK_SIZE, int BRICK_PADDING>
__global__ void sample_cache_init_first_kernel(const RaymarchInfo rm_info,
                                               const SceneInfo scene_info,
                                               const uint8_t *__restrict__ occupancy_grid,
                                               
                                               const int3 frustum_dims,
                                               const int3 brick_array_dims,
                                               BlockPayloadCacheInit* __restrict__ block_payloads,
                                               RayPayloadCacheInit* __restrict__ ray_payloads,
                                               
                                               const int* __restrict__ alive_block_idcs,
                                               int* __restrict__ subiter_block_idcs_compact,
                                               int* __restrict__ blocks_alive_buffer,
       
                                               int *subiter_n_steps_block)
{
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<Cache::PADDED_BRICK_SIZE*Cache::PADDED_BRICK_SIZE> tile = cg::tiled_partition<Cache::PADDED_BRICK_SIZE*Cache::PADDED_BRICK_SIZE>(block);

    constexpr int PADDED_BRICK_SIZE = BRICK_SIZE + 2 * BRICK_PADDING;
    if (threadIdx.x >= PADDED_BRICK_SIZE || threadIdx.y >= PADDED_BRICK_SIZE)
        return;

    // const int curr_subiter_block_idx = blockIdx.x;
    const int block_idx1D = blockIdx.x;

    const int2 ray_idx_in_block = make_int2(threadIdx.x, threadIdx.y);
    const int ray_id_in_block1D = ray_idx_in_block.y * PADDED_BRICK_SIZE + ray_idx_in_block.x;
    const int ray_id_global1D = block_idx1D * PADDED_BRICK_SIZE * PADDED_BRICK_SIZE + ray_id_in_block1D;

    const bool ray_is_padding = ray_idx_in_block.x < BRICK_PADDING || ray_idx_in_block.x >= (BRICK_SIZE + BRICK_PADDING) || 
                                ray_idx_in_block.y < BRICK_PADDING || ray_idx_in_block.y >= (BRICK_SIZE + BRICK_PADDING);

    if (!blocks_alive_buffer[block_idx1D])
        return;

    const int curr_subiter_block_idx = alive_block_idcs[blockIdx.x];

    BlockPayloadCacheInit block_payload = block_payloads[block_idx1D];
    RayPayloadCacheInit ray_payload = ray_payloads[ray_id_global1D];

    FrustumRay ray_world{ray_payload.origin, ray_payload.dir};

    int total_n_samples_interior = 0;
    int total_n_samples = 0;

    int n_alive_interior;
    while (true)
    {
        float t1 = rm_info.stepsize_info.t_from_step(block_payload.curr_block_step * BRICK_SIZE - BRICK_PADDING + rm_info.stepsize_info.near_i);

        int n_samples = 0;
        int n_samples_interior = 0;

        int is_alive_interior = (!ray_is_padding) * ray_payload.alive;
        n_alive_interior = cg::reduce(tile, is_alive_interior, cg::plus<int>());

        if (n_alive_interior <= 0)
            break;

        for (int step = 0; step < PADDED_BRICK_SIZE; step++)
        {
            bool is_padding = step < BRICK_PADDING || step >= (BRICK_SIZE + BRICK_PADDING);

            float t0 = t1;
            float dt = calculate_stepsize(t0, rm_info.stepsize_info);
            t1 = t0 + dt;
            float t_mid = (t0 + t1) * 0.5f;

            if (t_mid > ray_payload.far)
            {
                ray_payload.alive = false;
            }
            else
            {
                const float3 world_point = ray_world.at(t_mid);

                if (grid_occupied_at<DEFAULT_GRID_RESOLUTION, GRID_IS_MORTON>(world_point, scene_info, occupancy_grid))
                {
                    n_samples++;
                    n_samples_interior += !is_padding;
                }
            }
        }

        total_n_samples_interior = cg::reduce(tile, n_samples_interior, cg::plus<int>());
        if (total_n_samples_interior > 0)
        {
            total_n_samples = cg::reduce(tile, n_samples, cg::plus<int>());
            ray_payload.subiter_n_steps = n_samples;
            ray_payload.subiter_offset_in_block = cg::exclusive_scan(tile, n_samples);
            break;
        }

        block_payload.curr_block_step++;
    }
    
    int is_alive_interior = (!ray_is_padding) * ray_payload.alive;
    n_alive_interior = cg::reduce(tile, is_alive_interior, cg::plus<int>());
    if (ray_id_in_block1D == 0)
    {
        subiter_block_idcs_compact[curr_subiter_block_idx] = block_idx1D;
        subiter_n_steps_block[curr_subiter_block_idx] = total_n_samples;
        block_payloads[block_idx1D] = block_payload;
        blocks_alive_buffer[block_idx1D] = n_alive_interior > 0; //((BRICK_SIZE * BRICK_SIZE) / 20);
    }
    ray_payloads[ray_id_global1D] = ray_payload;
}

template <int BRICK_SIZE, int BRICK_PADDING>
__global__ void sample_cache_init_second_kernel(const RaymarchInfo rm_info,
                                                const SceneInfo scene_info,
                                                const uint8_t *__restrict__ occupancy_grid,

                                                const int3 frustum_dims,
                                                const int3 brick_array_dims,
                                                const int total_n_bricks,

                                                BlockPayloadCacheInit *__restrict__ block_payloads,
                                                RayPayloadCacheInit *__restrict__ ray_payloads,

                                                int *__restrict__ brick_isset_counter,
                                                float3 *__restrict__ sample_pos,
                                                float3 *__restrict__ sample_dir,
                                                float *__restrict__ sample_t0,

                                                const int *__restrict__ block_idcs_compact,
                                                const int *__restrict__ blocks_alive_buffer,

                                                const int *__restrict__ subiter_n_steps_block,
                                                const int *__restrict__ subiter_sample_offset_blocks)
{
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<Cache::PADDED_BRICK_SIZE*Cache::PADDED_BRICK_SIZE> tile = cg::tiled_partition<Cache::PADDED_BRICK_SIZE*Cache::PADDED_BRICK_SIZE>(block);
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    constexpr int PADDED_BRICK_SIZE = BRICK_SIZE + 2 * BRICK_PADDING;
    if (threadIdx.x >= PADDED_BRICK_SIZE || threadIdx.y >= PADDED_BRICK_SIZE)
        return;

    const int curr_subiter_block_idx = blockIdx.x;
    const int block_idx1D = block_idcs_compact[curr_subiter_block_idx];
    const int block_n_steps = subiter_n_steps_block[curr_subiter_block_idx];
    const int block_sample_offset = subiter_sample_offset_blocks[curr_subiter_block_idx];

    if (block_n_steps == 0)
        return;

    const int2 ray_idx_in_block = make_int2(threadIdx.x, threadIdx.y);
    const int ray_id_in_block1D = ray_idx_in_block.y * PADDED_BRICK_SIZE + ray_idx_in_block.x;
    const int ray_id_global1D = block_idx1D * PADDED_BRICK_SIZE * PADDED_BRICK_SIZE + ray_id_in_block1D;

    const bool ray_is_padding = ray_idx_in_block.x < BRICK_PADDING || ray_idx_in_block.x >= (BRICK_SIZE + BRICK_PADDING) || 
                                ray_idx_in_block.y < BRICK_PADDING || ray_idx_in_block.y >= (BRICK_SIZE + BRICK_PADDING);

    BlockPayloadCacheInit block_payload = block_payloads[block_idx1D];
    RayPayloadCacheInit ray_payload = ray_payloads[ray_id_global1D];

    FrustumRay ray_world{ray_payload.origin, ray_payload.dir};


    int subiter_ray_offset_in_block = ray_payload.subiter_offset_in_block;
    int subiter_ray_n_samples = ray_payload.subiter_n_steps;
    int ray_sample_counter = 0;

    float t1 = rm_info.stepsize_info.t_from_step(block_payload.curr_block_step * BRICK_SIZE - BRICK_PADDING + rm_info.stepsize_info.near_i);
    for (int step = 0; step < PADDED_BRICK_SIZE; step++)
    {
        bool is_padding = step < BRICK_PADDING || step >= (BRICK_SIZE + BRICK_PADDING);

        float t0 = t1;
        float dt = calculate_stepsize(t0, rm_info.stepsize_info);
        t1 = t0 + dt;
        float t_mid = (t0 + t1) * 0.5f;

        if (t_mid > ray_payload.far)
        {
            ray_payload.alive = false;
        }
        else
        {
            const float3 world_point = ray_world.at(t_mid);

            if (grid_occupied_at<DEFAULT_GRID_RESOLUTION, GRID_IS_MORTON>(world_point, scene_info, occupancy_grid))
            {
                float3 world_point_contracted = apply_contraction(world_point, scene_info);
                float3 raydir_norm = unit_to_01(normalize(ray_payload.dir));

                int subiter_sample_offset_global = block_sample_offset + subiter_ray_offset_in_block + ray_sample_counter;
                sample_pos[subiter_sample_offset_global] = world_point_contracted;
                sample_dir[subiter_sample_offset_global] = raydir_norm;
                sample_t0[subiter_sample_offset_global] = t0;
                
                ray_sample_counter++;
            }
        }
    }

    int total_n_steps = cg::reduce(tile, ray_sample_counter, cg::plus<int>());    
    if (ray_id_in_block1D == 0)
    {
        block_payloads[block_idx1D] = block_payload;
    }
    ray_payloads[ray_id_global1D] = ray_payload;
}

template <int BRICK_SIZE, int BRICK_PADDING>
__global__ void accumulateNew_cache_init_second_kernel(const RaymarchInfo rm_info,
                                                       const SceneInfo scene_info,
                                                       const CacheSettings cache_settings,

                                                       const int3 frustum_dims,
                                                       const int3 brick_array_dims,
                                                       const int total_n_bricks,

                                                       BlockPayloadCacheInit *__restrict__ block_payloads,
                                                       RayPayloadCacheInit *__restrict__ ray_payloads,

                                                       int batch_size,
                                                       const int latent_offset,
                                                       const half *__restrict__ network_output,
                                                       const float *__restrict__ sample_t0,

                                                       int *__restrict__ brick_isset_counter,
                                                       const int *__restrict__ block_idcs_compact,
                                                       int *__restrict__ blocks_alive_buffer,

                                                       const int *__restrict__ subiter_n_steps_block,
                                                       const int *__restrict__ subiter_sample_offset_blocks,
                                                       cudaSurfaceObject_t packed_brick_index_array_3D,
                                                       char *__restrict__ brick_isset_array,

                                                       const int3 data_array_dims,
                                                       const int3 data_array_bricks_per_dim,

                                                       Cache::Surfaces surfaces)
{
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<Cache::PADDED_BRICK_SIZE*Cache::PADDED_BRICK_SIZE> tile = cg::tiled_partition<Cache::PADDED_BRICK_SIZE*Cache::PADDED_BRICK_SIZE>(block);
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    constexpr int PADDED_BRICK_SIZE = BRICK_SIZE + 2 * BRICK_PADDING;
    if (threadIdx.x >= PADDED_BRICK_SIZE || threadIdx.y >= PADDED_BRICK_SIZE)
        return;

    // static constexpr CacheSettings cache_settings = CacheRenderer::default_cache_settings;

    const int curr_subiter_block_idx = blockIdx.x;
    const int block_idx1D = block_idcs_compact[curr_subiter_block_idx];
    const int2 block_idx2D = make_int2(block_idx1D % brick_array_dims.x, block_idx1D / brick_array_dims.x);
    const int block_n_steps = subiter_n_steps_block[curr_subiter_block_idx];
    const int block_sample_offset = subiter_sample_offset_blocks[curr_subiter_block_idx];

    if (block_n_steps == 0 || !blocks_alive_buffer[block_idx1D])
        return;

    const int2 ray_idx_in_block = make_int2(threadIdx.x, threadIdx.y);
    const int ray_id_in_block1D = ray_idx_in_block.y * PADDED_BRICK_SIZE + ray_idx_in_block.x;
    const int ray_id_global1D = block_idx1D * PADDED_BRICK_SIZE * PADDED_BRICK_SIZE + ray_id_in_block1D;
    const int2 actual_ray_idx = block_idx2D * BRICK_SIZE + ray_idx_in_block - BRICK_PADDING;

    const bool ray_is_padding = ray_idx_in_block.x < BRICK_PADDING || ray_idx_in_block.x >= (BRICK_SIZE + BRICK_PADDING) || 
                                ray_idx_in_block.y < BRICK_PADDING || ray_idx_in_block.y >= (BRICK_SIZE + BRICK_PADDING);

    BlockPayloadCacheInit block_payload = block_payloads[block_idx1D];
    RayPayloadCacheInit ray_payload = ray_payloads[ray_id_global1D];

    FrustumRay ray_world{ray_payload.origin, ray_payload.dir};

    int subiter_ray_offset_in_block = ray_payload.subiter_offset_in_block;
    int subiter_ray_n_samples = ray_payload.subiter_n_steps;
    int ray_sample_counter = 0;

    const int subiter_sample_offset_global = block_sample_offset + subiter_ray_offset_in_block;

    float transmittance = ray_payload.transmittance;

    int n_samples = 0;
    int n_visible_samples = 0;
    int n_visible_samples_interior = 0;

    float ray_t0 = subiter_ray_n_samples > 0 ? sample_t0[subiter_sample_offset_global] : 1e10f;
    float t1 = rm_info.stepsize_info.t_from_step(block_payload.curr_block_step * BRICK_SIZE - BRICK_PADDING + rm_info.stepsize_info.near_i);
    for (int step = 0; step < PADDED_BRICK_SIZE; step++)
    {
        bool is_padding = step < BRICK_PADDING || step >= (BRICK_SIZE + BRICK_PADDING);

        float t0 = t1;
        float dt = calculate_stepsize(t0, rm_info.stepsize_info);
        t1 = t0 + dt;
        float t_mid = (t0 + t1) * 0.5f;

        if (t_mid > ray_payload.far)
        {
            ray_payload.alive = false;
        }
        else if (fabs(ray_t0 - t0) < 1e-6f)
        {
            const float density = exp(__half2float(network_output[subiter_sample_offset_global + ray_sample_counter]));
            float alpha = clamp(1.0f - exp(-density * dt), 0.0f, 1.0f);

            n_samples++;

            if (alpha > scene_info.alpha_thre)
            {
                transmittance *= (is_padding ? 1.0f : (1.0f - alpha));

                n_visible_samples++;
                n_visible_samples_interior += !is_padding;
            }

            if (transmittance < 1e-4f)
            {
                ray_payload.alive = false;
            }

            ray_sample_counter++;
            ray_t0 = ray_sample_counter < subiter_ray_n_samples ? sample_t0[subiter_sample_offset_global + ray_sample_counter] : 1e10f;
        }
    }
    ray_payload.transmittance = transmittance;

    int total_n_visible_samples_interior = cg::reduce(tile, n_visible_samples_interior, cg::plus<int>());

    __shared__ int brick_data_index;
    __shared__ int overflow;

    if (ray_id_in_block1D == 0)
        overflow = 0;

    if (total_n_visible_samples_interior > 0)
    {
        const int3 brick_sup_idx3D = make_int3(block_idx2D.x, block_idx2D.y, block_payload.curr_block_step);
        int brick_idx = to1D(brick_sup_idx3D, brick_array_dims);

        if (ray_id_in_block1D == 0)
        {
            brick_data_index = atomicAdd(brick_isset_counter, 1);
            int3 brick_data_index3D = to3D(brick_data_index, data_array_bricks_per_dim);
            overflow = brick_data_index >= total_n_bricks;

            brick_isset_array[brick_idx] = !overflow;
            surf3Dwrite<int>(overflow ? -1 : pack3D(brick_data_index3D), packed_brick_index_array_3D, brick_sup_idx3D);
        }
        __syncthreads();

        int3 data_brick_sup_idx3D = to3D(brick_data_index, data_array_bricks_per_dim);

        if (!overflow)
        {
            block_payload.n_total_bricks += 1;

            ray_sample_counter = 0;
            float ray_t0 = subiter_ray_n_samples > 0 ? sample_t0[subiter_sample_offset_global] : 1e10f;
            float t1 = rm_info.stepsize_info.t_from_step(block_payload.curr_block_step * BRICK_SIZE - BRICK_PADDING + rm_info.stepsize_info.near_i);
            for (int step = 0; step < PADDED_BRICK_SIZE; step++)
            {
                bool is_padding = step < BRICK_PADDING || step >= (BRICK_SIZE + BRICK_PADDING);
                const int row_offset = subiter_sample_offset_global + ray_sample_counter;

                float t0 = t1;
                float dt = calculate_stepsize(t0, rm_info.stepsize_info);
                t1 = t0 + dt;
                float t_mid = (t0 + t1) * 0.5f;

                const int3 brick_sub_idx3D = make_int3(ray_idx_in_block.x - BRICK_PADDING, ray_idx_in_block.y - BRICK_PADDING, step - BRICK_PADDING);
                const int3 data_idx_3D = expand3D(data_brick_sup_idx3D, BRICK_SIZE, brick_sub_idx3D, BRICK_PADDING);

                float density = 0.0f;

                if (t_mid > ray_payload.far)
                {
                    ray_payload.alive = false;
                }
                else if (fabs(ray_t0 - t0) < 1e-6f)
                {
                    density = exp(__half2float(network_output[row_offset]));
                    float alpha = clamp(1.0f - exp(-density * dt), 0.0f, 1.0f);

                    bool overwrite_first_intermediate = false;
                    half first_intermediate_val;

                    switch (cache_settings.interpol_variant)
                    {
                        case DensityInterpolVariant::Density:
                        {
                            surf3Dwrite<float>(density, surfaces.data_density_alpha_surf, data_idx_3D);
                            break;
                        }
                        case DensityInterpolVariant::Alpha:
                        {
                            surf3Dwrite<float>(alpha, surfaces.data_density_alpha_surf, data_idx_3D);
                            break;
                        }
                        case DensityInterpolVariant::Intermediates:
                        {
                            break;
                        }
                        case DensityInterpolVariant::DensityIntermediates:
                        {
                            overwrite_first_intermediate = true;
                            first_intermediate_val = __float2half(density);
                            surf3Dwrite<float>(density, surfaces.data_density_alpha_surf, data_idx_3D); // store in density surface aswell (redundant)
                            break;
                        }
                        case DensityInterpolVariant::AlphaIntermediates:
                        {
                            overwrite_first_intermediate = true;
                            first_intermediate_val = __float2half(alpha);
                            surf3Dwrite<float>(alpha, surfaces.data_density_alpha_surf, data_idx_3D); // store in density surface aswell (redundant)
                            break;
                        }
                    }

                    surf3Dwrite<float>(1.0f, surfaces.data_isset_surf, data_idx_3D);

                    for (int i = 0; i < Cache::N_DATA_ARRAYS; i++)
                    {
                        const ushort4 tmp_val = make_ushort4(__half_as_short(i == 0 && overwrite_first_intermediate ? first_intermediate_val : network_output[row_offset + (latent_offset + i * 4 + 0) * batch_size]),
                                                            __half_as_ushort(network_output[row_offset + (latent_offset + i * 4 + 1) * batch_size]),
                                                            __half_as_ushort(network_output[row_offset + (latent_offset + i * 4 + 2) * batch_size]),
                                                            __half_as_ushort(network_output[row_offset + (latent_offset + i * 4 + 3) * batch_size]));

                        surf3Dwrite<ushort4>(tmp_val, surfaces.data_surf[i], data_idx_3D);
                    }

                    ray_sample_counter++;
                    ray_payload.total_n_steps += 1;
                    ray_t0 = ray_sample_counter < subiter_ray_n_samples ? sample_t0[row_offset+1] : 1e10f;
                }
            }
        }
    }

    int is_alive_interior = (!ray_is_padding) * ray_payload.alive;
    int n_alive_interior = cg::reduce(tile, is_alive_interior, cg::plus<int>());

    if (ray_id_in_block1D == 0)
    {
        // if overflow, kill at current start of block (termination depth)
        block_payload.curr_block_step += (overflow ? 0 : 1);
        block_payloads[block_idx1D] = block_payload;
        blocks_alive_buffer[block_idx1D] = !overflow && n_alive_interior > 0; //ceil((BRICK_SIZE * BRICK_SIZE) / 20);
    }
    ray_payloads[ray_id_global1D] = ray_payload;
}

template<int BRICK_SIZE, int BRICK_PADDING>
__global__ void writeImageBuffer_cache_init_kernel2(const int2 resolution,
                                                    const int3 brick_array_dims,
                                                    const RaymarchInfo rm_info, 
                                                    const BlockPayloadCacheInit *__restrict__ block_payloads,
                                                    const RayPayloadCacheInit *__restrict__ ray_payloads,
                                                    cudaSurfaceObject_t known_ranges_to, // TODO: Do somewhere else
                                                    cudaSurfaceObject_t image_buffer)
{
    constexpr int PADDED_CACHE_BLOCK_SIZE = BRICK_SIZE + 2 * BRICK_PADDING;
    if (threadIdx.x >= PADDED_CACHE_BLOCK_SIZE || threadIdx.y >= PADDED_CACHE_BLOCK_SIZE)
        return;

    const int block_idx1D = blockIdx.y * brick_array_dims.x + blockIdx.x;
    const int2 block_offset = make_int2(blockIdx.x, blockIdx.y) * BRICK_SIZE;

    const int2 ray_idx_in_block = make_int2(threadIdx.x, threadIdx.y);
    const int ray_id_in_block1D = ray_idx_in_block.y * PADDED_CACHE_BLOCK_SIZE + ray_idx_in_block.x;
    const int ray_id_global1D = block_idx1D * PADDED_CACHE_BLOCK_SIZE * PADDED_CACHE_BLOCK_SIZE + ray_id_in_block1D;

    const int2 pixel = block_offset + ray_idx_in_block - BRICK_PADDING;
    const bool is_padding = ray_idx_in_block.x < BRICK_PADDING || ray_idx_in_block.x >= (BRICK_SIZE + BRICK_PADDING) || 
                            ray_idx_in_block.y < BRICK_PADDING || ray_idx_in_block.y >= (BRICK_SIZE + BRICK_PADDING);

    BlockPayloadCacheInit block_payload = block_payloads[block_idx1D];

    if (!is_padding && pixel.x < resolution.x && pixel.y < resolution.y)
    {
        RayPayloadCacheInit payload = ray_payloads[ray_id_global1D];

        float termination_depth = rm_info.stepsize_info.t_from_step(block_payload.curr_block_step * BRICK_SIZE + rm_info.stepsize_info.near_i);
        surf2Dwrite(termination_depth, known_ranges_to, pixel.x * sizeof(float), pixel.y);

        float3 rgb = make_float3(min(block_payload.n_total_bricks / 32.f, 1.0f));
        float transmittance = 0.0f; // payload.transmittance;

        rgb = applyWhiteBackground(rgb, transmittance);
        surf2Dwrite(color_to_uchar4(rgb), image_buffer, pixel.x * sizeof(uchar4), pixel.y);
    }
}

__global__ void accumulate_cache_init_first_kernel(int n_alive, 
    int batch_size, 
    const RaymarchInfo rm_info, 
    const SceneInfo scene_info,
    const RayPayload* __restrict__ payloads,
    const half* __restrict__ network_output,
    const float* __restrict__ sample_t0,

    const int3 frustum_dims,
    const int3 brick_array_dims,
    char* __restrict__ brick_isset_array)
{
    const int curr_subiter_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (curr_subiter_idx >= n_alive)
        return;

    const RayPayload payload = payloads[curr_subiter_idx];
    const int2 ray_idx = to2DFlipped(payload.ray_id, rm_info.cam_info.resolution);
    float transmittance = payload.transmittance;

    // bool debug = true;
    // bool debug_ray = debug && payload.ray_id == (400 * rm_info.cam_info.resolution.x + 400);

    int step = 0;
    while (step < payload.subiter_n_steps_resample)
    {
        int row_offset = step * n_alive + curr_subiter_idx;
        const float density = exp(__half2float(network_output[row_offset]));

        const float t0 = sample_t0[row_offset];
        const float dt = calculate_stepsize(t0, rm_info.stepsize_info);
        const float t1 = t0 + dt;
        const float t_mid = (t0 + t1) / 2.0f;

        float alpha = clamp(1.0f - exp(-density * dt), 0.0f, 1.0f);

        if (alpha > scene_info.alpha_thre)
        {
            transmittance *= (1.0f - alpha);

            const int bin = __float2int_rd(rm_info.stepsize_info.step_from_t(t_mid) - rm_info.stepsize_info.near_i);

            const int3 idx3D = make_int3(ray_idx.x, ray_idx.y, bin);
            const int3 brick_idx3D = idx3D / Cache::BRICK_SIZE;
            const int brick_idx = to1D(brick_idx3D, brick_array_dims);
            brick_isset_array[brick_idx] = 1;
        }
        step++;

        if (transmittance < 1e-4f)
            break;
    }
}

__global__ void accumulate_cache_init_second_kernel(int n_alive, 
    int batch_size, 
    const RaymarchInfo rm_info, 
    const SceneInfo scene_info,
    const CacheSettings cache_settings,
    RayPayload* __restrict__ payloads,
    const half* __restrict__ network_output,
    const int latent_offset,
    const float* __restrict__ sample_t0,

    const int3 frustum_dims,
    const int3 brick_array_dims,
    const char* __restrict__ brick_isset_array,
    const int* __restrict__ brick_index_array,
    cudaTextureObject_t packed_brick_index_array_3D,

    const int3 data_array_dims,
    const int3 data_array_bricks_per_dim,
    
    Cache::Surfaces surfaces)
{
    const int curr_subiter_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (curr_subiter_idx >= n_alive)
        return;

    // static constexpr CacheSettings cache_settings = CacheRenderer::default_cache_settings;

    RayPayload payload = payloads[curr_subiter_idx];
    const int2 ray_idx = to2DFlipped(payload.ray_id, rm_info.cam_info.resolution);

    // bool debug = true;
    // bool debug_ray = debug && payload.ray_id == (542 * rm_info.cam_info.resolution.x + 785);

    int step = 0;
    while (step < payload.subiter_n_steps_resample)
    {
        int row_offset = step * n_alive + curr_subiter_idx;
        const float density = exp(min(__half2float(network_output[row_offset]), 10.0f));

        const float t0 = sample_t0[row_offset];
        const float dt = calculate_stepsize(t0, rm_info.stepsize_info);
        const float t1 = t0 + dt;
        const float t_mid = (t0 + t1) / 2.0f;

        const float alpha = clamp(1.0f - exp(-density * dt), 0.0f, 1.0f);

        if (alpha > scene_info.alpha_thre)
        {
            payload.transmittance *= (1.0f - alpha);

            int bin = __float2int_rd(rm_info.stepsize_info.step_from_t(t_mid) - rm_info.stepsize_info.near_i);

            const int3 idx3D = make_int3(ray_idx.x, ray_idx.y, bin);
            const int3 brick_sup_idx3D = idx3D / Cache::BRICK_SIZE;
            const int3 brick_sub_idx3D = idx3D - (brick_sup_idx3D * Cache::BRICK_SIZE);
            const int brick_idx1D = to1D(brick_sup_idx3D, brick_array_dims);

            // const int data_brick_idx1D = brick_index_array[brick_idx1D];
            // const int3 data_brick_sup_idx3D = to3D(data_brick_idx1D, data_array_bricks_per_dim);

            const int3 data_brick_sup_idx3D = unpack3D(tex3D<int>(packed_brick_index_array_3D, brick_sup_idx3D));
            const int3 data_idx_3D = expand3D(data_brick_sup_idx3D, Cache::BRICK_SIZE, brick_sub_idx3D, Cache::BRICK_PADDING);

            if (gridContains3D(data_idx_3D, data_array_dims))
            {
                float data_isset = surf3Dread<float>(surfaces.data_isset_surf, data_idx_3D.x * sizeof(float), data_idx_3D.y, data_idx_3D.z);
                assert(data_isset == 0.0f && "This data index should only be touched by this exact sample");
                    
                surf3Dwrite<float>(1.0f, surfaces.data_isset_surf, data_idx_3D);

                bool overwrite_first_intermediate = false;
                half first_intermediate_val;

                switch (cache_settings.interpol_variant)
                {
                    case DensityInterpolVariant::Density:
                    {
                        surf3Dwrite<float>(density, surfaces.data_density_alpha_surf, data_idx_3D);
                        break;
                    }
                    case DensityInterpolVariant::Alpha:
                    {
                        surf3Dwrite<float>(alpha, surfaces.data_density_alpha_surf, data_idx_3D);
                        break;
                    }
                    case DensityInterpolVariant::Intermediates:
                    {
                        break;
                    }
                    case DensityInterpolVariant::DensityIntermediates:
                    {
                        overwrite_first_intermediate = true;
                        first_intermediate_val = __float2half(density);
                        surf3Dwrite<float>(density, surfaces.data_density_alpha_surf, data_idx_3D);
                        break;
                    }
                    case DensityInterpolVariant::AlphaIntermediates:
                    {
                        overwrite_first_intermediate = true;
                        first_intermediate_val = __float2half(alpha);
                        surf3Dwrite<float>(alpha, surfaces.data_density_alpha_surf, data_idx_3D);
                        break;
                    }
                }


                for (int i = 0; i < Cache::N_DATA_ARRAYS; i++)
                {
                    const ushort4 tmp_val = make_ushort4(__half_as_short(i == 0 && overwrite_first_intermediate ? first_intermediate_val : network_output[row_offset + (latent_offset + i * 4 + 0) * batch_size]),
                                                         __half_as_ushort(network_output[row_offset + (latent_offset + i * 4 + 1) * batch_size]),
                                                         __half_as_ushort(network_output[row_offset + (latent_offset + i * 4 + 2) * batch_size]),
                                                         __half_as_ushort(network_output[row_offset + (latent_offset + i * 4 + 3) * batch_size]));

                    surf3Dwrite<ushort4>(tmp_val, surfaces.data_surf[i], data_idx_3D);
                }
            }
            else
            {
                // There is an overflow in the cache. Terminate the ray at the beginning of this bin. Will not accumulate to full opacity
                payload.alive = false;
                payload.termination_depth = t0;
                payload.cache_overflow = true;
                break;
            }            
        }

        step++;

        if (payload.transmittance < 1e-4f)
        {
            payload.alive = false;
            payload.termination_depth = t1;
            break;
        }
    }

    payloads[curr_subiter_idx] = payload;
}

__global__ void reserveNewBricks_kernel(const int3 brick_array_dims,
    const int3 data_array_bricks_per_dim,
    const int total_n_bricks, 

    int* __restrict__ brick_isset_counter,
    const char* __restrict__ brick_isset_array_subiter,
    char* __restrict__ brick_isset_array,
    int* __restrict__ brick_index_array,
    cudaSurfaceObject_t packed_brick_index_array_3D)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= brick_array_dims.x || y >= brick_array_dims.y)
        return;

    for (int z = 0; z < brick_array_dims.z; z++)
    {
        int brick_idx = to1D(make_int3(x, y, z), brick_array_dims);

        if (!brick_isset_array[brick_idx] && brick_isset_array_subiter[brick_idx])
        {
            int brick_data_index = atomicAdd(brick_isset_counter, 1);
            int3 brick_data_index3D = to3D(brick_data_index, data_array_bricks_per_dim);
            char overflow = brick_data_index >= total_n_bricks;

            brick_isset_array[brick_idx] = !overflow;
            brick_index_array[brick_idx] = brick_data_index;
            surf3Dwrite<int>(overflow ? -1 : pack3D(brick_data_index3D), packed_brick_index_array_3D, make_int3(x, y, z));
        }
    }
}

__global__ void writeImageBuffer_cache_init_kernel(const int2 resolution,
                                                   const RayPayload *__restrict__ final_payloads,
                                                   cudaSurfaceObject_t known_ranges_to, // TODO: Do somewhere else
                                                   cudaSurfaceObject_t image_buffer)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= resolution.x || y >= resolution.y)
        return;

    const int ray_id = y * resolution.x + x;

    RayPayload payload = final_payloads[ray_id];
    float3 rgb = make_float3(payload.transmittance < 1e-4f);
    float transmittance = 0.0f; // payload.transmittance;
    surf2Dwrite(payload.termination_depth, known_ranges_to, x * sizeof(float), y);

    rgb = applyWhiteBackground(rgb, transmittance);
    surf2Dwrite(color_to_uchar4(rgb), image_buffer, x * sizeof(uchar4), y);
}

CacheRenderer::CacheRenderer(SceneInfo scene_info, CudaBuffer<uint8_t> &occupancy_grid, NerfNetwork<half> &nerf_network, Cache* cache, Cache* next_cache)
    : Renderer(scene_info, occupancy_grid, nerf_network), _cache(cache), _cache_next(next_cache)
{
    int leastPriority=-1, greatestPriority=-1;
    cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
    cudaStreamCreateWithPriority(&_low_priority_stream, cudaStreamNonBlocking, leastPriority);
}

CacheRenderer::~CacheRenderer()
{
    cudaStreamDestroy(_low_priority_stream);
}

void CacheRenderer::resizeCache(int2 resolution, StepsizeInfo stepsize_info)
{
    // brick_sparsity = if a brick is occupied, then what precentage of the brick contains samples?
    //                = n_samples / (n_bricks_set * BRICK_SIZE^3);
    int n_rays = resolution.x * resolution.y;
    int estimated_n_samples = n_rays * Cache::INIT_ESTIMATED_N_SAMPLES_PER_RAY;
    int estimated_n_samples_per_brick = Cache::INIT_ESTIMATED_BRICK_SPARSITY * pow(Cache::BRICK_SIZE, 3);
    int estimated_n_bricks = estimated_n_samples / estimated_n_samples_per_brick;

    int max_n_steps = ceil(max_step_in_scene(stepsize_info, _scene_info));
    int frustum_depth_padded = divRoundUp(max_n_steps, 4) * 4; // pad in z-direction for alignment
    int3 frustum_dims = make_int3(resolution.x, resolution.y, frustum_depth_padded);

    _cache_next->resize(frustum_dims, estimated_n_bricks);
}

void CacheRenderer::initCache(RaymarchInfo rm_info, RenderBuffer& debug_image_buffer)
{
    resizeCache(rm_info.cam_info.resolution, rm_info.stepsize_info);
    resizeRenderbuffers(rm_info.cam_info.resolution);
    _cache_next->_rm_info = rm_info;

    _cache_next->setDataArraysZero();

    const int BLOCK_SIZE_1D = 128;
    const int BLOCK_SIZE_2D = 16;
    dim3 block_size2D(BLOCK_SIZE_2D, BLOCK_SIZE_2D);

    int total_brick_array_size = _cache_next->_brick_array_dims.x * _cache_next->_brick_array_dims.y * _cache_next->_brick_array_dims.z;

    int2 cache_array_bricks2D = make_int2(_cache_next->brickArrayDims().x, _cache_next->brickArrayDims().y);
    int3 cache_bricks_dims = _cache_next->dataArrayBricksPerDim();
    int total_n_possible_bricks = cache_bricks_dims.x * cache_bricks_dims.y * cache_bricks_dims.z;

    int n_rays = rm_info.cam_info.resolution.x * rm_info.cam_info.resolution.y;
    int n_alive = n_rays;


    initRayPayloads_common(rm_info, _scene_info, _payloads_doublebuffer.data());
    CUDA_SYNC_CHECK_THROW();

    int n_samples_evaluated_total = 0;

    int subiter = 0;
    while (subiter < MAX_SUBITER)
    {
        MY_CUDA_CHECK_THROW(cudaMemset(_cache_next->_brick_isset_array_subiter, 0, total_brick_array_size * sizeof(char)));

        RayPayload *prev_payloads = _payloads_doublebuffer.data() + (subiter % 2) * n_rays;
        RayPayload *curr_payloads = _payloads_doublebuffer.data() + ((subiter + 1) % 2) * n_rays;

        n_alive = compactPayloads_atomic_common(n_alive, prev_payloads, curr_payloads, _payloads_final.data(), _alive_counter.data());
        if (n_alive == 0)
            break;

        dim3 grid_size_alive_1D = divRoundUp(n_alive, BLOCK_SIZE_1D);

        int n_steps_curr_subiter = clamp(TARGET_BATCH_SIZE / n_alive, MIN_N_STEPS_PER_SUBITER, MAX_N_STEPS_PER_SUBITER);
        int curr_batch_size = tcnn::next_multiple(n_alive * n_steps_curr_subiter, (int)tcnn::batch_size_granularity);
        n_samples_evaluated_total += curr_batch_size;

        sample_common(n_alive, n_steps_curr_subiter, rm_info, _scene_info, _occupancy_grid.data(), curr_payloads, _samples_cache);

        tcnn::GPUMatrix<float> network_input_pos((float *)_samples_cache.pos.data(), 3, curr_batch_size);
        tcnn::GPUMatrix<float> network_input_dir((float *)_samples_cache.dir.data(), 3, curr_batch_size);
        tcnn::GPUMatrix<half, tcnn::RM> intermediates_out_matrix(_network_intermediate_buffer.data(), _nerf_network.padded_latent_width(), curr_batch_size);

        _nerf_network.inferenceLatents(cudaStreamDefault, network_input_pos, network_input_dir, intermediates_out_matrix);

        accumulate_cache_init_first_kernel<<<grid_size_alive_1D, BLOCK_SIZE_1D>>>(n_alive, curr_batch_size, rm_info, _scene_info,
                                                                                  curr_payloads, _network_intermediate_buffer.data(), _samples_cache.t0.data(),
                                                                                  _cache_next->frustumDims(), _cache_next->brickArrayDims(), _cache_next->_brick_isset_array_subiter);

        dim3 grid_size2D = toDim3(divRoundUp(cache_array_bricks2D, BLOCK_SIZE_2D));
        reserveNewBricks_kernel<<<grid_size2D, block_size2D>>>(_cache_next->brickArrayDims(), _cache_next->dataArrayBricksPerDim(), total_n_possible_bricks, _cache_next->_brick_isset_counter,
                                                               _cache_next->_brick_isset_array_subiter, _cache_next->_brick_isset_array, _cache_next->_brick_index_array, _cache_next->_packed_brick_index_array_3D.surface());

        accumulate_cache_init_second_kernel<<<grid_size_alive_1D, BLOCK_SIZE_1D>>>(n_alive, curr_batch_size, rm_info, _scene_info, cache_settings,
                                                                                   curr_payloads, _network_intermediate_buffer.data(), _nerf_network.latent_offset(), _samples_cache.t0.data(),
                                                                                   _cache_next->frustumDims(), _cache_next->brickArrayDims(), 
                                                                                   _cache_next->_brick_isset_array_subiter, _cache_next->_brick_index_array, _cache_next->_packed_brick_index_array_3D.texturePt(),
                                                                                   _cache_next->dataArrayDims(), _cache_next->dataArrayBricksPerDim(), _cache_next->surfaces());

        subiter++;
    }

    int brick_count;
    cudaMemcpy(&brick_count, _cache_next->_brick_isset_counter, sizeof(int), cudaMemcpyDeviceToHost);

    _init_stats.n_bricks_set = cubGetDeviceSum<char, int>(_cache_next->_brick_isset_array, total_brick_array_size);
    _init_stats.n_bricks_reserved = total_n_possible_bricks;
    _init_stats.samples_ppx = n_samples_evaluated_total / (float) n_rays;

    dim3 block_size_2D(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
    dim3 image_grid_size_2D(toDim3(divRoundUp(rm_info.cam_info.resolution, BLOCK_SIZE_2D)));
    writeImageBuffer_cache_init_kernel<<<image_grid_size_2D, block_size_2D>>>(rm_info.cam_info.resolution, _payloads_final.data(), _cache_next->_known_ranges_to.surface(), debug_image_buffer.surface());
}


void CacheRenderer::initCacheBlockwise(RaymarchInfo rm_info, RenderBuffer& debug_image_buffer)
{
    std::scoped_lock lk(_write_mutex);
    resizeCache(rm_info.cam_info.resolution, rm_info.stepsize_info);
    _cache_next->_rm_info = rm_info;

    _cache_next->setDataArraysZero(_low_priority_stream);

    int3 cache_bricks_dims = _cache_next->dataArrayBricksPerDim();
    int total_n_possible_bricks = cache_bricks_dims.x * cache_bricks_dims.y * cache_bricks_dims.z;
    int n_rays = rm_info.cam_info.resolution.x * rm_info.cam_info.resolution.y;
    int total_brick_array_size = _cache_next->_brick_array_dims.x * _cache_next->_brick_array_dims.y * _cache_next->_brick_array_dims.z;

    constexpr int PADDED_BRICK_SIZE = Cache::BRICK_SIZE + 2 * Cache::BRICK_PADDING;
    dim3 grid_size_2D(toDim3(divRoundUp(rm_info.cam_info.resolution, Cache::BRICK_SIZE)));
    dim3 block_size_2D(PADDED_BRICK_SIZE, PADDED_BRICK_SIZE);
    int total_n_brick2D = grid_size_2D.x * grid_size_2D.y;

    if (_block_payloads_init.size() != total_n_brick2D)
    {
        _ray_payloads_init.resize(total_n_brick2D * PADDED_BRICK_SIZE * PADDED_BRICK_SIZE);
        _block_payloads_init.resize(total_n_brick2D);
        _blocks_alive_buffer.resize(total_n_brick2D);
        _alive_block_idcs.resize(total_n_brick2D);
        _subiter_block_idcs_compact.resize(total_n_brick2D);
        _subiter_n_steps_blocks.resize(total_n_brick2D);
        _subiter_sample_offset_blocks.resize(total_n_brick2D);

        _samples_cache_init.resize(total_n_brick2D * PADDED_BRICK_SIZE * PADDED_BRICK_SIZE * PADDED_BRICK_SIZE);
        _network_intermediate_buffer_init.resize(total_n_brick2D * PADDED_BRICK_SIZE * PADDED_BRICK_SIZE * PADDED_BRICK_SIZE * _nerf_network.required_buffer_width());

        _cub_alive_cache_buffer_result.resize(1);

        size_t temp_storage_bytes;
        CUDA_CHECK_THROW(cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes, _blocks_alive_buffer.data(), _alive_block_idcs.data(), total_n_brick2D, _low_priority_stream));
        size_t temp_storage_type_size = sizeof(decltype(_cub_alive_cache_buffer_tmp)::type);
        _cub_alive_cache_buffer_tmp.resize(round_up_pow2(temp_storage_bytes, temp_storage_type_size) / temp_storage_type_size);

        CUDA_CHECK_THROW(cub::DeviceReduce::Sum(nullptr, temp_storage_bytes, _blocks_alive_buffer.data(), _cub_alive_cache_buffer_result.data(), total_n_brick2D, _low_priority_stream));
        temp_storage_type_size = sizeof(decltype(_cub_alive_cache_reduce_buffer_tmp)::type);
        _cub_alive_cache_reduce_buffer_tmp.resize(round_up_pow2(temp_storage_bytes, temp_storage_type_size) / temp_storage_type_size);
    }
    // CUDA_SYNC_CHECK_THROW_ASYNC(_low_priority_stream);
    
    // std::cout << "Before Init: " << grid_size_2D.x << ", " << grid_size_2D.y << " | " << block_size_2D.x << ", " << block_size_2D.y << std::endl;
    initPayloads_cache_init_kernel<Cache::BRICK_SIZE, Cache::BRICK_PADDING><<<grid_size_2D, block_size_2D, 0, _low_priority_stream>>>(rm_info, _scene_info, _cache_next->frustumDims(), _cache_next->brickArrayDims(), 
        _block_payloads_init.data(), _ray_payloads_init.data(), _blocks_alive_buffer.data());
    // CUDA_SYNC_CHECK_THROW_ASYNC(_low_priority_stream);

    size_t temp_storage_bytes = _cub_alive_cache_buffer_tmp.sizeInBytes();
    size_t temp_storage_reduce_bytes = _cub_alive_cache_reduce_buffer_tmp.sizeInBytes();

    int subiter = 0;
    while (subiter < MAX_SUBITER)
    {
        CUDA_CHECK_THROW(cub::DeviceScan::ExclusiveSum(_cub_alive_cache_buffer_tmp.data(), temp_storage_bytes, _blocks_alive_buffer.data(), _alive_block_idcs.data(), total_n_brick2D, _low_priority_stream));

        int n_blocks_alive;
        CUDA_CHECK_THROW(cub::DeviceReduce::Sum(_cub_alive_cache_reduce_buffer_tmp.data(), temp_storage_reduce_bytes, _blocks_alive_buffer.data(), _cub_alive_cache_buffer_result.data(), total_n_brick2D, _low_priority_stream));
        CUDA_CHECK_THROW(cudaMemcpyAsync(&n_blocks_alive, _cub_alive_cache_buffer_result.data(), sizeof(int), cudaMemcpyDeviceToHost, _low_priority_stream));
        CUDA_SYNC_CHECK_THROW_ASYNC(_low_priority_stream);

        if (n_blocks_alive == 0)
            break;

        sample_cache_init_first_kernel<Cache::BRICK_SIZE, Cache::BRICK_PADDING><<<total_n_brick2D, block_size_2D, 0, _low_priority_stream>>>(rm_info, _scene_info, 
            _occupancy_grid.data(), _cache_next->frustumDims(), _cache_next->brickArrayDims(), 
            _block_payloads_init.data(), _ray_payloads_init.data(), 
            _alive_block_idcs.data(), _subiter_block_idcs_compact.data(), _blocks_alive_buffer.data(), _subiter_n_steps_blocks.data()); 
        // CUDA_SYNC_CHECK_THROW_ASYNC(_low_priority_stream);

        int total_n_samples_compact;
        CUDA_CHECK_THROW(cub::DeviceReduce::Sum(_cub_alive_cache_reduce_buffer_tmp.data(), temp_storage_reduce_bytes, _subiter_n_steps_blocks.data(), _cub_alive_cache_buffer_result.data(), n_blocks_alive, _low_priority_stream));
        CUDA_CHECK_THROW(cudaMemcpyAsync(&total_n_samples_compact, _cub_alive_cache_buffer_result.data(), sizeof(int), cudaMemcpyDeviceToHost, _low_priority_stream));

        CUDA_CHECK_THROW(cub::DeviceScan::ExclusiveSum(_cub_alive_cache_buffer_tmp.data(), temp_storage_bytes, _subiter_n_steps_blocks.data(), _subiter_sample_offset_blocks.data(), n_blocks_alive, _low_priority_stream));
        // CUDA_SYNC_CHECK_THROW_ASYNC(_low_priority_stream);

        // std::cout << "Subiter: " << subiter << ", N alive: " << n_blocks_alive << ", N samples: " << total_n_samples_compact << " vs. " << total_n_samples_compact_old << "(Max=" << _samples_cache_init.dir.size() << ")" << std::endl;
        sample_cache_init_second_kernel<Cache::BRICK_SIZE, Cache::BRICK_PADDING><<<n_blocks_alive, block_size_2D, 0, _low_priority_stream>>>(rm_info, _scene_info, 
            _occupancy_grid.data(), _cache_next->frustumDims(), _cache_next->brickArrayDims(), total_n_brick2D, 
            _block_payloads_init.data(), _ray_payloads_init.data(), _cache_next->_brick_isset_counter,
            _samples_cache_init.pos.data(), _samples_cache_init.dir.data(), _samples_cache_init.t0.data(),
            _subiter_block_idcs_compact.data(), _blocks_alive_buffer.data(), _subiter_n_steps_blocks.data(), _subiter_sample_offset_blocks.data()); 
        // CUDA_SYNC_CHECK_THROW_ASYNC(_low_priority_stream);

        if (total_n_samples_compact <= 0)
            break;

        int curr_batch_size = tcnn::next_multiple(total_n_samples_compact, (int)tcnn::batch_size_granularity);

        tcnn::GPUMatrix<float> network_input_pos((float *)_samples_cache_init.pos.data(), 3, curr_batch_size);
        tcnn::GPUMatrix<float> network_input_dir((float *)_samples_cache_init.dir.data(), 3, curr_batch_size);
        tcnn::GPUMatrix<half, tcnn::RM> intermediates_out_matrix(_network_intermediate_buffer_init.data(), _nerf_network.padded_latent_width(), curr_batch_size);

        _nerf_network.inferenceLatents(_low_priority_stream, network_input_pos, network_input_dir, intermediates_out_matrix);
        // CUDA_SYNC_CHECK_THROW_ASYNC(_low_priority_stream);

        accumulateNew_cache_init_second_kernel<Cache::BRICK_SIZE, Cache::BRICK_PADDING><<<n_blocks_alive, block_size_2D, 0, _low_priority_stream>>>(rm_info, _scene_info, cache_settings,
            _cache_next->frustumDims(), _cache_next->brickArrayDims(), total_n_possible_bricks, _block_payloads_init.data(), _ray_payloads_init.data(), 
            curr_batch_size, _nerf_network.latent_offset(), _network_intermediate_buffer_init.data(), _samples_cache_init.t0.data(),
            _cache_next->_brick_isset_counter, _subiter_block_idcs_compact.data(), _blocks_alive_buffer.data(), _subiter_n_steps_blocks.data(), _subiter_sample_offset_blocks.data(),
            _cache_next->_packed_brick_index_array_3D.surface(), _cache_next->_brick_isset_array, _cache_next->dataArrayDims(), _cache_next->dataArrayBricksPerDim(), _cache_next->surfaces());
        // CUDA_SYNC_CHECK_THROW_ASYNC(_low_priority_stream);

        subiter++;    
    }

    writeImageBuffer_cache_init_kernel2<Cache::BRICK_SIZE, Cache::BRICK_PADDING><<<grid_size_2D, block_size_2D, 0, _low_priority_stream>>>(rm_info.cam_info.resolution, _cache_next->brickArrayDims(), rm_info,
        _block_payloads_init.data(), _ray_payloads_init.data(), _cache_next->_known_ranges_to.surface(), debug_image_buffer.surface());

    int brick_count;
    cudaMemcpy(&brick_count, _cache_next->_brick_isset_counter, sizeof(int), cudaMemcpyDeviceToHost);
    CUDA_SYNC_CHECK_THROW_ASYNC(_low_priority_stream);

    _init_stats.n_bricks_set = cubGetDeviceSum<char, int>(_cache_next->_brick_isset_array, total_brick_array_size, _low_priority_stream);
    _init_stats.n_bricks_reserved = total_n_possible_bricks;
    _init_stats.samples_ppx = 0.0f; //n_samples_evaluated_total / (float) n_rays;

    // std::cout << "Bricks set: " << brick_count << ", " << _init_stats.n_bricks_set << std::endl;
}

void CacheRenderer::swapCache()
{
    std::scoped_lock lk(_read_mutex, _write_mutex);
    std::swap(_cache, _cache_next);
}