/*
 * Copyright (C) 2024, Michael Steiner, Graz University of Technology.
 * This code is licensed under the MIT license.
 */

#include "renderer.h"

#include "raymarch_common.h"
#include "util/cub_helper.h"

#include "util/random_val.h"

__global__ void createMortonMask_kernel(const int2 actual_resolution, const int2 next_power2_resolution, int* is_in_actual_resolution)
{
    const uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;
    
    is_in_actual_resolution[morton2D(x, y)] = (x >= actual_resolution.x || y >= actual_resolution.y);
}
__global__ void createMortonRayOrder_kernel(const int2 actual_resolution, const int* morton_offset_correction, int* ray_order)
{
    const uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= actual_resolution.x || y >= actual_resolution.y)
        return;

    const int linear_ray_id = y * actual_resolution.x + x;

    const int morton_id = morton2D(x, y);
    const int corrected_morton_id = morton_id - morton_offset_correction[morton_id];

    ray_order[linear_ray_id] = corrected_morton_id;
}

__global__ void initRayPayloads_common_kernel(const RaymarchInfo rm_info,
                                              const SceneInfo scene_info,
                                              RayPayload* __restrict__ payloads, 
                                              const int* __restrict__ ray_order_buffer)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= rm_info.cam_info.resolution.x || y >= rm_info.cam_info.resolution.y)
        return;

    const int ray_id = y * rm_info.cam_info.resolution.x + x;
    const int ray_order = (ray_order_buffer != nullptr) ? ray_order_buffer[ray_id] : ray_id;

    float2 pixel_offset = rm_info.deterministic ? make_float2(0.5f, 0.5f) : ld_random_pixel_offset(rm_info.sample_index);

    const float2 screen_point = make_float2(x + pixel_offset.x, y + pixel_offset.y);
    const FrustumRay ray = generateRay(ray_id, screen_point, rm_info);

    float aabb_t0, aabb_t1;
    bool hits_aabb = ray_aabb_intersect(ray, scene_info.aabb_from, scene_info.aabb_to, aabb_t0, aabb_t1);

    float start_t = hits_aabb ? valid_t_from_t(aabb_t0, rm_info.stepsize_info) : rm_info.stepsize_info.near;
    float start_dt = calculate_stepsize(start_t, rm_info.stepsize_info);
    float t_offset = start_dt * (rm_info.deterministic ? 0.0f : ld_random_t_offset(rm_info.sample_index, ray_id * 56924617));

    RayPayload payload{
        ray.origin,
        ray.dir,
        ray_id,
        hits_aabb ? min(aabb_t1, rm_info.stepsize_info.far) : rm_info.stepsize_info.far,

        hits_aabb,
        start_t + t_offset,
        0,
        0,

        0.0f,
        0.0f,
        0,
        0,
        {0.0f, 0.0f, 0.0f},
        1.0f,
        rm_info.stepsize_info.far,
        false
    };

    payloads[ray_order] = payload;
}

__global__ void initRayPayloads_split_kernel(const RaymarchInfo rm_info,
                                             const SceneInfo scene_info,
                                             BaseRayInitInfo *__restrict__ inits,
                                             SegmentedRayPayload *__restrict__ payloads,
                                             BaseRayResult *__restrict__ results,
                                             int *alive_buffer,
                                             const int* ray_order_buffer)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= rm_info.cam_info.resolution.x || y >= rm_info.cam_info.resolution.y)
        return;

    const int ray_id = y * rm_info.cam_info.resolution.x + x;
    const int ray_order = (ray_order_buffer != nullptr) ? ray_order_buffer[ray_id] : ray_id;

    float2 pixel_offset = rm_info.deterministic ? make_float2(0.5f, 0.5f) : ld_random_pixel_offset(rm_info.sample_index);

    const float2 screen_point = make_float2(x + pixel_offset.x, y + pixel_offset.y);
    const FrustumRay ray = generateRay(ray_id, screen_point, rm_info);

    float aabb_t0, aabb_t1;
    bool hits_aabb = ray_aabb_intersect(ray, scene_info.aabb_from, scene_info.aabb_to, aabb_t0, aabb_t1);

    inits[ray_order] = {
        origin : ray.origin,
        dir : ray.dir,
        ray_id : ray_id,
        far : hits_aabb ? min(aabb_t1, rm_info.stepsize_info.far) : rm_info.stepsize_info.far,
        //near_i : rm_info.stepsize_info.near_i needed if using SAMPLE_WITH_RTX
    };

    // Designated initialize doesn't work for struct inheritance
    float start_t = hits_aabb ? valid_t_from_t(aabb_t0, rm_info.stepsize_info) : rm_info.stepsize_info.near;
    float start_dt = calculate_stepsize(start_t, rm_info.stepsize_info);
    float t_offset = start_dt * (rm_info.deterministic ? 0.0f : ld_random_t_offset(rm_info.sample_index, ray_id * 56924617));
    payloads[ray_order] = {
        curr_t1 : start_t + t_offset,
        curr_segment_end : 0.0f,
    };

    results[ray_order] = {
        rgb : {0.0f, 0.0f, 0.0f},
        transmittance : 1.0f
    };

    alive_buffer[ray_order] = hits_aabb;
}

__global__ void compactPayloads_atomic_common_kernel(int n_alive,
                                              const RayPayload *__restrict__ prev_payloads,
                                              RayPayload *__restrict__ curr_payloads,
                                              RayPayload *__restrict__ final_payloads,
                                              int *__restrict__ alive_counter)
{
    const int curr_subiter_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (curr_subiter_idx >= n_alive)
        return;

    RayPayload prev_payload = prev_payloads[curr_subiter_idx];
    if (prev_payload.alive)
    {
        int new_subiter_idx = atomicAdd(alive_counter, 1);
        curr_payloads[new_subiter_idx] = prev_payload;
    }
    else
    {
        final_payloads[prev_payload.ray_id] = prev_payload;
    }
}

__global__ void compactPayloads_split_kernel(int n_alive,
                                             const int *__restrict__ new_index_buffer,
                                             const int *__restrict__ alive_buffer,
                                             const BaseRayInitInfo *__restrict__ prev_inits,
                                             const SegmentedRayPayload *__restrict__ prev_payloads,
                                             const BaseRayResult *__restrict__ prev_results,
                                             BaseRayInitInfo *__restrict__ curr_inits,
                                             SegmentedRayPayload *__restrict__ curr_payloads,
                                             BaseRayResult *__restrict__ curr_results,
                                             BaseRayResult *__restrict__ final_results)
{
    const int curr_subiter_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (curr_subiter_idx >= n_alive)
        return;

    const BaseRayInitInfo prev_init = prev_inits[curr_subiter_idx];
    const SegmentedRayPayload prev_payload = prev_payloads[curr_subiter_idx];
    const BaseRayResult prev_result = prev_results[curr_subiter_idx];

    int new_idx = new_index_buffer[curr_subiter_idx] - 1;

    if (alive_buffer[curr_subiter_idx])
    {
        curr_inits[new_idx] = prev_init;
        curr_payloads[new_idx] = prev_payload;
        curr_results[new_idx] = prev_result;
    }
    else
    {
        final_results[prev_init.ray_id] = prev_result;
    }
}


__global__ void sample_common_kernel(int n_alive,
                                     int n_steps_curr_subiter,
                                     RaymarchInfo rm_info,
                                     SceneInfo scene_info,
                                     const uint8_t* __restrict__ occupancy_grid,
                                     RayPayload* __restrict__ payloads,
                                     float3* __restrict__ sample_pos,
                                     float3* __restrict__ sample_dir,
                                     float* __restrict__ sample_t0)
{
    const int curr_subiter_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (curr_subiter_idx >= n_alive)
        return;

    int step = 0;
    RayPayload payload = payloads[curr_subiter_idx];

    float t1 = payload.curr_t1;
    while (step < n_steps_curr_subiter)
    {
        float t0 = t1;
        float dt = calculate_stepsize(t0, rm_info.stepsize_info);
        t1 = t0 + dt;
        float t_mid = (t0 + t1) * 0.5f;

        if (t_mid > payload.far)
        {
            payload.alive = false;
            break;
        }

        const float3 world_point = payload.origin + t_mid * payload.dir;
        if (grid_occupied_at<DEFAULT_GRID_RESOLUTION, GRID_IS_MORTON>(world_point, scene_info, occupancy_grid))
        {
            float3 world_point_contracted = apply_contraction(world_point, scene_info);
            float3 raydir_norm = unit_to_01(normalize(payload.dir));

            int row_offset = step * n_alive + curr_subiter_idx;
            sample_pos[row_offset] = world_point_contracted;
            sample_dir[row_offset] = raydir_norm;
            sample_t0[row_offset] = t0;

            step++;
        }
    }

    payload.curr_t1 = t1;
    payload.subiter_n_steps_resample = step;

    payloads[curr_subiter_idx] = payload;
}

__global__ void accumulate_common_kernel(int n_alive,
                                         int batch_size,
                                         const RaymarchInfo rm_info,
                                         const SceneInfo scene_info,
                                         RayPayload* __restrict__ payloads,
                                         const half* __restrict__ network_output_rgbs,
                                         const float* __restrict__ sample_t0)
{
    const int curr_subiter_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (curr_subiter_idx >= n_alive)
        return;

    int step = 0;
    RayPayload payload = payloads[curr_subiter_idx];

    float3 tmp_network_output_rgb;
    float tmp_network_output_density;
    while (step < payload.subiter_n_steps_resample)
    {
        int row_offset = step * n_alive + curr_subiter_idx;
        tmp_network_output_rgb.x = __half2float(network_output_rgbs[row_offset + 0 * batch_size]);
        tmp_network_output_rgb.y = __half2float(network_output_rgbs[row_offset + 1 * batch_size]);
        tmp_network_output_rgb.z = __half2float(network_output_rgbs[row_offset + 2 * batch_size]);
        tmp_network_output_density = __half2float(network_output_rgbs[row_offset + 3 * batch_size]);

        const float3 rgb = sigmoid(tmp_network_output_rgb);
        const float density = exp(tmp_network_output_density);

        const float t0 = sample_t0[row_offset];
        const float dt = calculate_stepsize(t0, rm_info.stepsize_info);
        const float t1 = t0 + dt;

        float alpha = clamp(1.0f - exp(-density * dt), 0.0f, 1.0f);

        if (alpha > scene_info.alpha_thre)
        {
            float weight = alpha * payload.transmittance;
            payload.rgb += weight * rgb;
            payload.transmittance *= (1.0f - alpha);

            payload.total_n_steps_resample += 1;
            payload.total_contrib_resample += weight;
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


__global__ void writeImageBuffer_common_kernel(const int2 resolution,
                                               const RayPayload *__restrict__ final_payloads,
                                               cudaSurfaceObject_t image_buffer)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= resolution.x || y >= resolution.y)
        return;

    const int ray_id = y * resolution.x + x;

    RayPayload payload = final_payloads[ray_id];

    // float3 rgb = make_float3(clamp(payload.termination_depth / 16.f, 0.0f, 1.0f));
    // float transmittance = 0.0f;

    float3 rgb = payload.rgb;
    float transmittance = payload.transmittance;

    rgb = applyWhiteBackground(rgb, transmittance);
    surf2Dwrite(color_to_uchar4(rgb), image_buffer, x * sizeof(uchar4), y);
}

void createRayOrder(const int2 resolution, CudaBuffer<int>& ray_order_buffer)
{
    int2 next_power2_res = make_int2(pow(2, ceil(log2(max(resolution.x, resolution.y)))));
    int n_elems_next_power2 = next_power2_res.x * next_power2_res.y;

    CudaBuffer<int> morton_offset_correction_buffer(n_elems_next_power2);

    constexpr int BLOCK_SIZE_2D = 16;
    dim3 block_size_2D(BLOCK_SIZE_2D, BLOCK_SIZE_2D);

    dim3 grid_size_power2_2D(toDim3(divRoundUp(next_power2_res, BLOCK_SIZE_2D)));
    createMortonMask_kernel<<<grid_size_power2_2D, block_size_2D>>>(resolution, next_power2_res, morton_offset_correction_buffer.data());
    cubDeviceExclusiveSum(morton_offset_correction_buffer.data(), morton_offset_correction_buffer.data(), n_elems_next_power2);

    dim3 grid_size_actual_2D(toDim3(divRoundUp(resolution, BLOCK_SIZE_2D)));
    createMortonRayOrder_kernel<<<grid_size_actual_2D, block_size_2D>>>(resolution, morton_offset_correction_buffer.data(), ray_order_buffer.data());
}

void initRayPayloads_common(const RaymarchInfo rm_info, const SceneInfo scene_info, RayPayload *payloads, const int* ray_order_buffer)
{
    constexpr int BLOCK_SIZE_2D = 16;
    dim3 block_size_2D(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
    dim3 grid_size_2D(toDim3(divRoundUp(rm_info.cam_info.resolution, BLOCK_SIZE_2D)));
    initRayPayloads_common_kernel<<<grid_size_2D, block_size_2D>>>(rm_info, scene_info, payloads, ray_order_buffer);
}

void initRayPayloads_split(RaymarchInfo &rm_info, SceneInfo &scene_info, RayBuffer<BaseRayInitInfo, SegmentedRayPayload, BaseRayResult>& ray_buffer, int *alive_buffer, const int* ray_order_buffer)
{
    constexpr int BLOCK_SIZE_2D = 16;
    dim3 block_size_2D(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
    dim3 grid_size_2D(toDim3(divRoundUp(rm_info.cam_info.resolution, BLOCK_SIZE_2D)));
    initRayPayloads_split_kernel<<<grid_size_2D, block_size_2D>>>(rm_info, scene_info, ray_buffer.init_infos, ray_buffer.payloads, ray_buffer.results, alive_buffer, ray_order_buffer);
}

int compactPayloads_atomic_common(int n_alive, const RayPayload *prev_payloads,
                                  RayPayload *curr_payloads, RayPayload *final_payloads, 
                                  int *alive_counter)
{
    cudaMemsetAsync(alive_counter, 0, sizeof(int));
    
    tcnn::linear_kernel(compactPayloads_atomic_common_kernel, 0, cudaStreamDefault,
                        n_alive, prev_payloads, curr_payloads,
                        final_payloads, alive_counter);

    // CUDA_SYNC_CHECK_THROW();
    cudaMemcpyAsync(&n_alive, alive_counter, sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    return n_alive;
}

int compactPayloads_coherent_split(int n_alive, int* new_index_buffer, int *alive_buffer, 
                                   RayBuffer<BaseRayInitInfo, SegmentedRayPayload, BaseRayResult> prev_rays,
                                   RayBuffer<BaseRayInitInfo, SegmentedRayPayload, BaseRayResult> curr_rays,
                                   BaseRayResult *final_results)
{
    int n_alive_new; // Last index is equal to the number of elements
    cudaMemcpyAsync(&n_alive_new, new_index_buffer + (n_alive - 1), sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    tcnn::linear_kernel(compactPayloads_split_kernel, 0, cudaStreamDefault,
                        n_alive, new_index_buffer, alive_buffer,
                        prev_rays.init_infos, prev_rays.payloads, prev_rays.results,
                        curr_rays.init_infos, curr_rays.payloads, curr_rays.results, final_results);

    return n_alive_new;
}


void sample_common(int n_alive, int n_steps_curr_subiter, const RaymarchInfo rm_info,
                   const SceneInfo scene_info, const uint8_t *occupancy_grid,
                   RayPayload *payloads, SampleInfoBuffer &sample_buffer)
{
    const int BLOCK_SIZE = 128;
    dim3 grid_size_alive_1D = divRoundUp(n_alive, BLOCK_SIZE);
    sample_common_kernel<<<grid_size_alive_1D, BLOCK_SIZE>>>(n_alive, n_steps_curr_subiter, rm_info, scene_info, occupancy_grid,
                                                             payloads, sample_buffer.pos.data(), sample_buffer.dir.data(), sample_buffer.t0.data());
}

void accumulate_common(int n_alive, int batch_size, const RaymarchInfo rm_info, const SceneInfo scene_info,
                       RayPayload *payloads, const half *network_output_rgbs, SampleInfoBuffer &sample_buffer)
{
    const int BLOCK_SIZE = 128;
    dim3 grid_size_alive_1D = divRoundUp(n_alive, BLOCK_SIZE);
    accumulate_common_kernel<<<grid_size_alive_1D, BLOCK_SIZE>>>(n_alive, batch_size, rm_info, scene_info,
                                                                 payloads, network_output_rgbs, sample_buffer.t0.data());
}

void writeImageBuffer_common(const int2 resolution, const RayPayload *final_payloads, RenderBuffer &image_buffer, DebugData* debug_data)
{
    constexpr int BLOCK_SIZE_2D = 16;
    dim3 block_size_2D(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
    dim3 grid_size_2D(toDim3(divRoundUp(resolution, BLOCK_SIZE_2D)));
    writeImageBuffer_common_kernel<<<grid_size_2D, block_size_2D>>>(resolution, final_payloads, image_buffer.surface());
}