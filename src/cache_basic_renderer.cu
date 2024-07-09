
#include "cache_basic_renderer.h"

#include "raymarch_common.h"
#include "util/cub_helper.h"


void inline __device__ setNetworkIntermediates(half* network_input, const float4 vals, int row_offset, int batch_size, int i)
{
    network_input[row_offset + (i * 4 + 0) * batch_size] = __float2half(vals.x);
    network_input[row_offset + (i * 4 + 1) * batch_size] = __float2half(vals.y);
    network_input[row_offset + (i * 4 + 2) * batch_size] = __float2half(vals.z);
    network_input[row_offset + (i * 4 + 3) * batch_size] = __float2half(vals.w);
}

__global__ void sample_cache_kernel(int n_alive,
                                    int n_steps_curr_subiter,
                                    int batch_size,

                                    const RaymarchInfo rm_info,
                                    const RaymarchInfo cache_rm_info,
                                    const CacheSettings cache_settings,

                                    const cudaTextureObject_t known_ranges_to,
                                    const char *froxel_brick_isset_array,
                                    const int *froxel_brick_index_array,
                                    const int3 froxel_grid_bricks_per_dims,
                                    const int3 froxel_grid_dims,

                                    const int3 data_array_bricks_per_dim,
                                    Cache::Textures data_array_textures,

                                    const SceneInfo scene_info,
                                    const uint8_t *__restrict__ occupancy_grid,
                                    RayPayload *__restrict__ payloads,

                                    float3 *__restrict__ sample_resample_pos,
                                    float3 *__restrict__ sample_resample_dir,
                                    float *__restrict__ sample_resample_t0,

                                    float3 *__restrict__ sample_cache_pos,
                                    float3 *__restrict__ sample_cache_dir,
                                    float3 *__restrict__ sample_init_viewdir,
                                    float *__restrict__ sample_cache_t0,
                                    float *__restrict__ samples_cache_density,

                                    half *__restrict__ mlp_head_network_inputs,
                                    const int latent_offset)
{
    const int curr_subiter_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (curr_subiter_idx >= n_alive)
        return;

    // static constexpr CacheSettings cache_settings = CacheRenderer::default_cache_settings;

    int step_cache = 0;
    int step_resample = 0;
    RayPayload payload = payloads[curr_subiter_idx];

    FrustumRay ray_world{payload.origin, payload.dir};
    FrustumRay ray_cache = transformRay(ray_world, cache_rm_info.cam_info.world2cam);

    // bool debug = true;
    // bool debug_ray = debug && payload.ray_id == (542 * rm_info.cam_info.resolution.x + 785);

    float transmittance = payload.transmittance;
    float t1 = payload.curr_t1;
    while (step_cache < n_steps_curr_subiter && step_resample < n_steps_curr_subiter)
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

        float froxel_t;
        const float3 world_point = ray_world.at(t_mid);
        const float3 cache_viewdir = normalize(world_point - cache_rm_info.cam_info.cam2world.getTranslation());

        if (grid_occupied_at<DEFAULT_GRID_RESOLUTION, GRID_IS_MORTON>(world_point, scene_info, occupancy_grid))
        {
            const float3 cache_cam_point = ray_cache.at(t_mid);
            const float3 froxel_point = cam2froxel(cache_cam_point, cache_rm_info.cam_info, cache_rm_info.stepsize_info, froxel_t);

            float2 cache_pixel_known_range = make_float2(cache_rm_info.stepsize_info.near, tex2D<float>(known_ranges_to, froxel_point.x, froxel_point.y));
            float3 world_point_contracted = apply_contraction(world_point, scene_info);
            float3 raydir_norm = unit_to_01(normalize(ray_world.dir));

            bool in_cache_fov = gridContainsXY(froxel_point, froxel_grid_dims);
            bool inside_grid = in_cache_fov && inRange(froxel_point.z, 0, froxel_grid_dims.z);
            bool in_unsampled_region = !in_cache_fov || froxel_t < cache_pixel_known_range.x || froxel_t > cache_pixel_known_range.y;

            if (!inside_grid || in_unsampled_region)
            {
                int row_offset_resample = step_resample * n_alive + curr_subiter_idx;

                sample_resample_pos[row_offset_resample] = world_point_contracted;
                sample_resample_dir[row_offset_resample] = raydir_norm;
                sample_resample_t0[row_offset_resample] = t0;

                step_resample++;
                continue;
            }

            bool inside_brick;
            float3 data_array_point;
            if (inside_grid && froxel2data<Cache::BRICK_SIZE>(froxel_point, froxel_brick_isset_array, froxel_brick_index_array, froxel_grid_bricks_per_dims,
                                                                  data_array_bricks_per_dim, cache_settings.use_inter_brick_interpolation, data_array_point, inside_brick))
            {
                float data_isset = tex3D<float>(data_array_textures.data_isset_tex_pt, data_array_point);

                if (data_isset > 0.0f)
                {
                    float4 cache_intermediate_vals = fetchBrickTex3DInterpol<Cache::BRICK_SIZE, float4>(data_array_textures.data_tex_pt[0], data_array_textures.data_tex_linear[0],
                                                                                                            data_array_point, inside_brick, froxel_point, froxel_brick_isset_array, froxel_brick_index_array, cache_settings.use_inter_brick_interpolation,
                                                                                                            froxel_grid_dims, froxel_grid_bricks_per_dims, data_array_bricks_per_dim, cache_settings.interpol_function);

                    const float weight = fetchBrickTex3DInterpol<Cache::BRICK_SIZE, float>(data_array_textures.data_isset_tex_pt, data_array_textures.data_isset_tex_linear,
                                                                                               data_array_point, inside_brick, froxel_point, froxel_brick_isset_array, froxel_brick_index_array, cache_settings.use_inter_brick_interpolation,
                                                                                               froxel_grid_dims, froxel_grid_bricks_per_dims, data_array_bricks_per_dim, cache_settings.interpol_function);

                    const float density_alpha_cache = fetchBrickTex3DInterpol<Cache::BRICK_SIZE, float>(data_array_textures.data_density_alpha_tex_pt, data_array_textures.data_density_alpha_tex_linear,
                                                                                                            data_array_point, inside_brick, froxel_point, froxel_brick_isset_array, froxel_brick_index_array, cache_settings.use_inter_brick_interpolation,
                                                                                                            froxel_grid_dims, froxel_grid_bricks_per_dims, data_array_bricks_per_dim, cache_settings.interpol_function);

                    float alpha, density;
                    switch (cache_settings.interpol_variant)
                    {
                        case DensityInterpolVariant::Density:
                        {
                            density = density_alpha_cache;
                            alpha = clamp(1.0f - exp(-density * dt), 0.0f, 1.0f);
                            break;
                        }
                        case DensityInterpolVariant::Alpha:
                        {
                            alpha = density_alpha_cache;
                            density = -log(max(1.0f - alpha, 1e-6f)) / dt;
                            break;
                        }
                        case DensityInterpolVariant::Intermediates:
                        {
                            density = exp(cache_intermediate_vals.x / weight);
                            alpha = clamp(1.0f - exp(-density * dt), 0.0f, 1.0f);
                            break;
                        }
                        case DensityInterpolVariant::DensityIntermediates:
                        {
                            density = cache_intermediate_vals.x;
                            alpha = clamp(1.0f - exp(-density * dt), 0.0f, 1.0f);
                            cache_intermediate_vals.x = log(density); 
                            break;
                        }
                        case DensityInterpolVariant::AlphaIntermediates:
                        {
                            alpha = cache_intermediate_vals.x;
                            density = -log(max(1.0f - alpha, 1e-6f)) / dt;
                            cache_intermediate_vals.x = log(density); 
                            break;
                        }
                    } 

                    if (cache_settings.reweight_intermediates && cache_settings.disable_reweighting_first_intermediate)
                        cache_intermediate_vals.x *= weight;

                    
                    if (alpha > scene_info.alpha_thre)
                    {
                        int row_offset_cache = step_cache * n_alive + curr_subiter_idx;

                        for (int i = 0; i < Cache::N_DATA_ARRAYS; i++)
                        {
                            if (i > 0)
                                cache_intermediate_vals = fetchBrickTex3DInterpol<Cache::BRICK_SIZE, float4>(data_array_textures.data_tex_pt[i], data_array_textures.data_tex_linear[i], data_array_point,
                                                                                                                 inside_brick, froxel_point, froxel_brick_isset_array, froxel_brick_index_array, cache_settings.use_inter_brick_interpolation,
                                                                                                                 froxel_grid_dims, froxel_grid_bricks_per_dims, data_array_bricks_per_dim, cache_settings.interpol_function);

                            float4 weighted_intermediate_vals = cache_intermediate_vals / (cache_settings.reweight_intermediates ? weight : 1.0f);
                            setNetworkIntermediates(mlp_head_network_inputs + latent_offset * batch_size, weighted_intermediate_vals, row_offset_cache, batch_size, i);
                        }

                        float3 world_point_contracted = apply_contraction(world_point, scene_info);
                        float3 raydir_norm = unit_to_01(normalize(ray_world.dir));

                        sample_cache_pos[row_offset_cache] = world_point_contracted;
                        sample_cache_dir[row_offset_cache] = raydir_norm;
                        sample_cache_t0[row_offset_cache] = t0;
                        samples_cache_density[row_offset_cache] = density;
                        sample_init_viewdir[row_offset_cache] = unit_to_01(cache_viewdir);

                        step_cache++;
                        transmittance *= (1.0f - alpha);

                        if (transmittance < 1e-4f)
                        {
                            payload.alive = false;
                            payload.termination_depth = t1;
                            break;
                        }
                    }
                }
            }
        }
    }

    payload.curr_t1 = t1;
    payload.subiter_n_steps_cache = step_cache;
    payload.subiter_n_steps_resample = step_resample;

    payloads[curr_subiter_idx] = payload;
}

__global__ void accumulate_cache_render_kernel(int n_alive,
                                               int batch_size,
                                               const RaymarchInfo rm_info,
                                               const SceneInfo scene_info,
                                               RayPayload *__restrict__ payloads,
                                               const half *__restrict__ network_output_rgbs_resample,
                                               const half *__restrict__ network_output_rgbs_cache,
                                               const float *__restrict__ sample_t0_resample,
                                               const float *__restrict__ sample_t0_cache,
                                               const float *__restrict__ samples_cache_density)
{
    const int curr_subiter_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (curr_subiter_idx >= n_alive)
        return;

    int step_resample = 0;
    int step_cache = 0;
    RayPayload payload = payloads[curr_subiter_idx];

    while (step_resample < payload.subiter_n_steps_resample || step_cache < payload.subiter_n_steps_cache)
    {
        int row_offset_resample = step_resample * n_alive + curr_subiter_idx;
        int row_offset_cache = step_cache * n_alive + curr_subiter_idx;

        const float t0_resample = step_resample < payload.subiter_n_steps_resample ? sample_t0_resample[row_offset_resample] : 1e10f;
        const float t0_cache = step_cache < payload.subiter_n_steps_cache ? sample_t0_cache[row_offset_cache] : 1e10f;

        float t0 = min(t0_resample, t0_cache);
        const float dt = calculate_stepsize(t0, rm_info.stepsize_info);
        const float t1 = t0 + dt;

        float3 tmp_network_output_rgb;
        float tmp_network_output_density;
        if (t0_resample < t0_cache)
        {
            tmp_network_output_rgb.x = __half2float(network_output_rgbs_resample[row_offset_resample + 0 * batch_size]);
            tmp_network_output_rgb.y = __half2float(network_output_rgbs_resample[row_offset_resample + 1 * batch_size]);
            tmp_network_output_rgb.z = __half2float(network_output_rgbs_resample[row_offset_resample + 2 * batch_size]);
            tmp_network_output_density = __half2float(network_output_rgbs_resample[row_offset_resample + 3 * batch_size]);

            step_resample++;
        }
        else
        {
            tmp_network_output_rgb.x = __half2float(network_output_rgbs_cache[row_offset_cache + 0 * batch_size]);
            tmp_network_output_rgb.y = __half2float(network_output_rgbs_cache[row_offset_cache + 1 * batch_size]);
            tmp_network_output_rgb.z = __half2float(network_output_rgbs_cache[row_offset_cache + 2 * batch_size]);
            
            // tmp_network_output_density = __half2float(network_output_rgbs_cache[row_offset_cache + 3 * batch_size]);
            tmp_network_output_density = log(samples_cache_density[row_offset_cache]); // Currently used so "Alpha" and "Density" Interpol Variants work

            step_cache++;
        }

        const float3 rgb = sigmoid(tmp_network_output_rgb);

        float density = exp(tmp_network_output_density);
        float alpha = clamp(1.0f - exp(-density * dt), 0.0f, 1.0f);

        if (alpha > scene_info.alpha_thre)
        {
            float weight = alpha * payload.transmittance;
            payload.rgb += weight * rgb;
            payload.transmittance *= (1.0f - alpha);

            if (t0_resample < t0_cache)
            {
                payload.total_n_steps_resample += 1;
                payload.total_contrib_resample += weight;
            }
            else
            {
                payload.total_n_steps_cache += 1;
                payload.total_contrib_cache += weight;
            }
        }

        if (payload.transmittance < 1e-4f)
        {
            payload.alive = false;
            payload.termination_depth = t1;
            break;
        }
    }

    payloads[curr_subiter_idx] = payload;
}

__global__ void writeImageBuffer_cacheBasic_postRender_kernel(const int2 resolution,
                                                              const RayPayload *__restrict__ final_payloads,
                                                              int *__restrict__ n_samples_contrib_resample,
                                                              int *__restrict__ n_samples_contrib_cache,
                                                              cudaSurfaceObject_t image_buffer)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= resolution.x || y >= resolution.y)
        return;

    const int ray_id = y * resolution.x + x;

    RayPayload payload = final_payloads[ray_id];

    n_samples_contrib_resample[ray_id] = payload.total_n_steps_resample;
    n_samples_contrib_cache[ray_id] = payload.total_n_steps_cache;

    float3 rgb = make_float3(payload.total_n_steps_cache > 0);
    surf2Dwrite(color_to_uchar4(rgb), image_buffer, x * sizeof(uchar4), y);
}

CacheBasicRenderer::CacheBasicRenderer(SceneInfo scene_info, CudaBuffer<uint8_t> &occupancy_grid, NerfNetwork<half> &nerf_network, Cache* cache, Cache* next_cache)
    : CacheRenderer(scene_info, occupancy_grid, nerf_network, cache, next_cache)
{
}

void CacheBasicRenderer::resizeRenderbuffers(int2 resolution)
{
    int n_rays = resolution.x * resolution.y;
    int buffer_size = tcnn::next_multiple(max(n_rays, TARGET_BATCH_SIZE) * MIN_N_STEPS_PER_SUBITER, (int)tcnn::batch_size_granularity);

    if (_force_renderbuffer_resize || n_rays > _payloads_final.size())
    {
        // if caching resolution and render resolution are different, keep size to the larger one
        _alive_counter.resize(1);

        _payloads_doublebuffer.resize(n_rays * 2);
        _payloads_final.resize(n_rays);

        _n_samples_contrib_resample.resize(n_rays);
        _n_samples_contrib_cache.resize(n_rays);

        _samples_resample.resize(buffer_size);
        _network_output_rgbs_resample.resize(buffer_size * _nerf_network.padded_output_width());

        _samples_cache.resize(buffer_size);
        _samples_cache_density.resize(buffer_size);
        _samples_cache_init_viewdir.resize(buffer_size);
        _network_intermediate_buffer.resize(buffer_size * _nerf_network.required_buffer_width());
        _network_output_rgbs_cache.resize(buffer_size * _nerf_network.padded_output_width());

        _force_renderbuffer_resize = false;
    }
}

void CacheBasicRenderer::render(RaymarchInfo rm_info, RenderBuffer& image_buffer, DebugData* debug_data)
{
    if (!_cache->_initialized)
        throw std::runtime_error("Cache is not initialized!");

    resizeRenderbuffers(rm_info.cam_info.resolution);

    int n_rays = rm_info.cam_info.resolution.x * rm_info.cam_info.resolution.y;
    int n_alive = n_rays;

    constexpr int BLOCK_SIZE_1D = 128;
    const int BLOCK_SIZE_2D = 16;
    dim3 grid_size_alive_1D(divRoundUp(n_alive, BLOCK_SIZE_1D));

    initRayPayloads_common(rm_info, _scene_info, _payloads_doublebuffer.data());

    _render_stats.total_n_evaluated_samples_resample = 0;
    _render_stats.total_n_contributing_samples_resample = 0;
    _render_stats.total_n_evaluated_samples_cache = 0;
    _render_stats.total_n_contributing_samples_cache = 0;

    int subiter = 0;
    while (subiter < MAX_SUBITER)
    {
        RayPayload *prev_payloads = _payloads_doublebuffer.data() + (subiter % 2) * n_rays;
        RayPayload *curr_payloads = _payloads_doublebuffer.data() + ((subiter + 1) % 2) * n_rays;

        n_alive = compactPayloads_atomic_common(n_alive, prev_payloads, curr_payloads, _payloads_final.data(), _alive_counter.data());
        if (n_alive == 0)
            break;

        grid_size_alive_1D = divRoundUp(n_alive, BLOCK_SIZE_1D);

        int n_steps_curr_subiter = clamp(TARGET_BATCH_SIZE / n_alive, MIN_N_STEPS_PER_SUBITER, MAX_N_STEPS_PER_SUBITER);
        int curr_batch_size = tcnn::next_multiple(n_alive * n_steps_curr_subiter, (int)tcnn::batch_size_granularity);
        _render_stats.total_n_evaluated_samples_resample += curr_batch_size;
        _render_stats.total_n_evaluated_samples_cache += curr_batch_size;

        sample_cache_kernel<<<grid_size_alive_1D, BLOCK_SIZE_1D>>>(n_alive, n_steps_curr_subiter, curr_batch_size, rm_info, _cache->_rm_info, cache_settings, 
            _cache->_known_ranges_to.texturePt(), _cache->_brick_isset_array, _cache->_brick_index_array, _cache->brickArrayDims(), _cache->frustumDims(), _cache->dataArrayBricksPerDim(), _cache->textures(), 
            _scene_info, _occupancy_grid.data(), curr_payloads,
            _samples_resample.pos.data(), _samples_resample.dir.data(), _samples_resample.t0.data(),
            _samples_cache.pos.data(), _samples_cache.dir.data(), _samples_cache_init_viewdir.data(), _samples_cache.t0.data(), _samples_cache_density.data(), 
            _network_intermediate_buffer.data(), _nerf_network.latent_offset());

        tcnn::GPUMatrix<float> network_input_pos_cache((float *)_samples_cache.pos.data(), 3, curr_batch_size);
        tcnn::GPUMatrix<float> network_input_dir_cache((float *)_samples_cache.dir.data(), 3, curr_batch_size);
        tcnn::GPUMatrix<float> network_input_init_viewdir_cache((float *)_samples_cache_init_viewdir.data(), 3, curr_batch_size);
        tcnn::GPUMatrix<half, tcnn::RM> mlp_head_input_matrix(_network_intermediate_buffer.data(), _nerf_network.m_mlp_head->input_width(), curr_batch_size);
        tcnn::GPUMatrix<half, tcnn::RM> rgbsigma_matrix_cache(_network_output_rgbs_cache.data(), _nerf_network.padded_output_width(), curr_batch_size);

        _nerf_network.inferenceHead(cudaStreamDefault, network_input_pos_cache, network_input_dir_cache, network_input_init_viewdir_cache, mlp_head_input_matrix, rgbsigma_matrix_cache);


        tcnn::GPUMatrix<float> network_input_pos_resample((float *)_samples_resample.pos.data(), 3, curr_batch_size);
        tcnn::GPUMatrix<float> network_input_dir_resample((float *)_samples_resample.dir.data(), 3, curr_batch_size);

        tcnn::GPUMatrix<half, tcnn::RM> rgbsigma_matrix_resample(_network_output_rgbs_resample.data(), _nerf_network.padded_output_width(), curr_batch_size);

        _nerf_network.inference(cudaStreamDefault, network_input_pos_resample, network_input_dir_resample, network_input_dir_resample, mlp_head_input_matrix, rgbsigma_matrix_resample);

        accumulate_cache_render_kernel<<<grid_size_alive_1D, BLOCK_SIZE_1D>>>(n_alive, curr_batch_size, rm_info, _scene_info,
                                                                              curr_payloads, _network_output_rgbs_resample.data(), _network_output_rgbs_cache.data(),
                                                                              _samples_resample.t0.data(), _samples_cache.t0.data(), _samples_cache_density.data());

        subiter++;
    }

    // dim3 block_size_2D(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
    // dim3 image_grid_size_2D(toDim3(divRoundUp(rm_info.cam_info.resolution, BLOCK_SIZE_2D)));
    // writeImageBuffer_cache_kernel<<<image_grid_size_2D, block_size_2D>>>(rm_info.cam_info.resolution, _payloads_final.data(), image_buffer.surface());

    writeImageBuffer_common(rm_info.cam_info.resolution, _payloads_final.data(), image_buffer);
}

void CacheBasicRenderer::postRender(RaymarchInfo rm_info, RenderBuffer& additional_image_buffer)
{
    if (!_cache->_initialized)
        throw std::runtime_error("Cache is not initialized!");
    
    const int BLOCK_SIZE_2D = 16;
    dim3 block_size_2D(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
    dim3 image_grid_size_2D(toDim3(divRoundUp(rm_info.cam_info.resolution, BLOCK_SIZE_2D)));
    writeImageBuffer_cacheBasic_postRender_kernel<<<image_grid_size_2D, block_size_2D>>>(rm_info.cam_info.resolution, _payloads_final.data(),
                                                                                         _n_samples_contrib_resample.data(), _n_samples_contrib_cache.data(), additional_image_buffer.surface());

    _render_stats.total_n_contributing_samples_resample = cubGetDeviceSum<int, int>(_n_samples_contrib_resample.data(), _n_samples_contrib_resample.size());
    _render_stats.total_n_contributing_samples_cache = cubGetDeviceSum<int, int>(_n_samples_contrib_cache.data(), _n_samples_contrib_cache.size());
}