
#include "performance_renderer.h"

#include "raymarch_common.h"
#include "util/cub_helper.h"

#ifdef RTX_ENABLED
#include "util/owl_helper.h"
#endif

__global__ void sample_performance_kernel(int n_alive,
                                          int n_steps_curr_subiter,
                                          const RaymarchInfo rm_info,
                                          const SceneInfo scene_info,
                                          const uint8_t *__restrict__ occupancy_grid,
                                          const BaseRayInitInfo *__restrict__ ray_inits,
                                          SegmentedRayPayload *__restrict__ payloads,
                                          int *__restrict__ alive_buffer,
                                          int *__restrict__ subiter_n_steps,
                                          float3 *__restrict__ sample_pos,
                                          float3 *__restrict__ sample_dir,
                                          float *__restrict__ sample_t0)
{
    const int curr_subiter_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (curr_subiter_idx >= n_alive)
        return;

    int step = 0;
    const BaseRayInitInfo ray_init = ray_inits[curr_subiter_idx];
    SegmentedRayPayload payload = payloads[curr_subiter_idx];

    bool alive = true;
    float t1 = payload.curr_t1;

    while (step < n_steps_curr_subiter)
    {
        float t0 = t1;
        float dt = calculate_stepsize(t0, rm_info.stepsize_info);
        t1 = t0 + dt;
        float t_mid = (t0 + t1) * 0.5f;

        if (t_mid > ray_init.far)
        {
            alive = false;
            break;
        }

        const float3 world_point = ray_init.origin + t_mid * ray_init.dir;
        if (grid_occupied_at<DEFAULT_GRID_RESOLUTION, GRID_IS_MORTON>(world_point, scene_info, occupancy_grid))
        {
            int row_offset = step * n_alive + curr_subiter_idx;
            sample_pos[row_offset] = apply_contraction(world_point, scene_info);
            sample_dir[row_offset] = unit_to_01(ray_init.dir);
            sample_t0[row_offset] = t0;

            step++;
        }
    }

    alive_buffer[curr_subiter_idx] = alive;
    subiter_n_steps[curr_subiter_idx] = step;

    payload.curr_t1 = t1;
    payloads[curr_subiter_idx] = payload;
}

__global__ void accumulate_performance_kernel(int n_alive,
                                              int batch_size,
                                              const RaymarchInfo rm_info,
                                              const SceneInfo scene_info,
                                              BaseRayResult *__restrict__ results,
                                              const int *__restrict__ subiter_n_steps,
                                              int *__restrict__ alive_buffer,
                                              const half *__restrict__ network_output_rgbs,
                                              const float *__restrict__ sample_t0)
{
    const int curr_subiter_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (curr_subiter_idx >= n_alive)
        return;

    int step = 0;
    BaseRayResult result = results[curr_subiter_idx];

    const int n_steps = subiter_n_steps[curr_subiter_idx];
    bool alive = alive_buffer[curr_subiter_idx];

    float3 tmp_network_output_rgb;
    float tmp_network_output_density;
    while (step < n_steps)
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

        float alpha = clamp(1.0f - exp(-density * dt), 0.0f, 1.0f);
        if (alpha > scene_info.alpha_thre)
        {
            float weight = alpha * result.transmittance;
            result.rgb += weight * rgb;
            result.transmittance *= (1.0f - alpha);
        }

        step++;
        if (result.transmittance < 1e-4f)
        {
            alive = false;
            break;
        }
    }

    results[curr_subiter_idx] = result;
    alive_buffer[curr_subiter_idx] = alive;
}

__global__ void writeImageBuffer_performance_kernel(const int2 resolution,
                                                    const BaseRayResult *__restrict__ final_results,
                                                    cudaSurfaceObject_t image_buffer,
                                                    const int sample_index)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= resolution.x || y >= resolution.y)
        return;

    const int ray_id = y * resolution.x + x;

    const BaseRayResult result = final_results[ray_id];
    float transmittance = result.transmittance;
    float3 rgb = applyWhiteBackground(result.rgb, transmittance);

    uchar4 rgb_8_prev;
    surf2Dread(&rgb_8_prev, image_buffer, x * sizeof(uchar4), y);

    uchar4 rgb_8_new = color_to_uchar4((rgb + uchar4_to_color(rgb_8_prev) * sample_index) / float(sample_index + 1));
    surf2Dwrite(rgb_8_new, image_buffer, x * sizeof(uchar4), y);
}

void writeImageBuffer_performance(const int2 resolution, const BaseRayResult *final_results, RenderBuffer &image_buffer, const int sample_index)
{
    constexpr int BLOCK_SIZE_2D = 16;
    dim3 block_size_2D(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
    dim3 grid_size_2D(toDim3(divRoundUp(resolution, BLOCK_SIZE_2D)));
    writeImageBuffer_performance_kernel<<<grid_size_2D, block_size_2D>>>(resolution, final_results, image_buffer.surface(), sample_index);
}

PerformanceRenderer::PerformanceRenderer(SceneInfo scene_info, CudaBuffer<uint8_t> &occupancy_grid, NerfNetwork<half> &nerf_network)
    : Renderer(scene_info, occupancy_grid, nerf_network)
{
    if (scene_info.grid_resolution != DEFAULT_GRID_RESOLUTION)
        throw std::runtime_error("Performance Renderer only supports fixed grid resolution!");

#ifdef RTX_ENABLED
    if (INIT_RAYS_WITH_RTX)
    {
        std::cout << "Initializing OWL..." << std::endl;
        _context = owlContextCreate(nullptr, 1);
        OWLGeomType triangle_geom_type = createGeomType(_context);

        createMeshFromGridCompact(_occupancy_grid_mesh, scene_info, occupancy_grid, _context);
        _grid_geom = buildAccel(_occupancy_grid_mesh, triangle_geom_type, _context);

        _init_rays_module = owlModuleCreate(_context, init_ray_payloads_ptx);
        _init_rays_program.init(_context, _init_rays_module, triangle_geom_type);

        owlBuildPrograms(_context);
        owlBuildPipeline(_context);

        _init_rays_program.dummyLaunch(_context); // to not have initial startup costs in render function
    }
#endif
}

void PerformanceRenderer::resizeRenderbuffers(int2 resolution)
{
    int n_rays = resolution.x * resolution.y;
    if (n_rays != _alive_buffer.size())
    {
        _ray_order_buffer.resize(n_rays);
        createRayOrder(resolution, _ray_order_buffer);
    }

    _ray_dblbuffer.resize(n_rays);
    _ray_results_final.resize(n_rays);

    int buffer_size = tcnn::next_multiple(max(n_rays, TARGET_BATCH_SIZE) * MIN_N_STEPS_PER_SUBITER, (int)tcnn::batch_size_granularity);

    _samples.resize(buffer_size);
    _network_intermediate_buffer.resize(buffer_size * _nerf_network.required_buffer_width());
    _network_output_rgbs.resize(buffer_size * _nerf_network.padded_output_width());

    _alive_buffer.resize(n_rays);
    _new_index_buffer.resize(n_rays);
    _subiter_n_steps.resize(n_rays);

    size_t temp_storage_bytes;
    CUDA_CHECK_THROW(cub::DeviceScan::InclusiveSum(nullptr, temp_storage_bytes, _alive_buffer.data(), _new_index_buffer.data(), n_rays));
    constexpr size_t temp_storage_type_size = sizeof(decltype(_cub_alive_buffer_tmp)::type);
    _cub_alive_buffer_tmp.resize(round_up_pow2(temp_storage_bytes, temp_storage_type_size) / temp_storage_type_size);
}

void PerformanceRenderer::render(RaymarchInfo rm_info, RenderBuffer &image_buffer, DebugData *debug_data)
{
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

    int n_samples_evaluated_total = 0;

    int subiter = 0;
    while (subiter < MAX_SUBITER)
    {
        auto prev_rays = _ray_dblbuffer.getPrevBuffer();
        auto curr_rays = _ray_dblbuffer.getCurrBuffer();

        _timer.start(TimingState::CompactRays);
        size_t tmp_size_bytes = _cub_alive_buffer_tmp.sizeInBytes();
        CUDA_CHECK_THROW(cub::DeviceScan::InclusiveSum(_cub_alive_buffer_tmp.data(), tmp_size_bytes, _alive_buffer.data(), _new_index_buffer.data(), n_alive));
        n_alive = compactPayloads_coherent_split(n_alive, _new_index_buffer.data(), _alive_buffer.data(), prev_rays, curr_rays, _ray_results_final.data());
        if (n_alive == 0)
            break;

        int n_steps_curr_subiter = clamp(TARGET_BATCH_SIZE / n_alive, MIN_N_STEPS_PER_SUBITER, MAX_N_STEPS_PER_SUBITER);
        int curr_batch_size = tcnn::next_multiple(n_alive * n_steps_curr_subiter, (int)tcnn::batch_size_granularity);
        n_samples_evaluated_total += curr_batch_size;

        _timer.syncElapsed();
        _timer.start(TimingState::Sample);
        tcnn::linear_kernel(sample_performance_kernel, 0, cudaStreamDefault,
                            n_alive, n_steps_curr_subiter, rm_info, _scene_info, _occupancy_grid.data(),
                            curr_rays.init_infos, curr_rays.payloads, _alive_buffer.data(), _subiter_n_steps.data(),
                            _samples.pos.data(), _samples.dir.data(), _samples.t0.data());

        tcnn::GPUMatrix<float> network_input_pos((float *)_samples.pos.data(), 3, curr_batch_size);
        tcnn::GPUMatrix<float> network_input_dir((float *)_samples.dir.data(), 3, curr_batch_size);

        tcnn::GPUMatrix<half, tcnn::RM> mlp_head_input_matrix(_network_intermediate_buffer.data(), _nerf_network.m_mlp_head->input_width(), curr_batch_size);
        tcnn::GPUMatrix<half, tcnn::RM> rgbsigma_matrix(_network_output_rgbs.data(), _nerf_network.padded_output_width(), curr_batch_size);

        _timer.start(TimingState::InferenceResample);
        _nerf_network.inference(cudaStreamDefault, network_input_pos, network_input_dir, network_input_dir, mlp_head_input_matrix, rgbsigma_matrix);

        _timer.start(TimingState::Accumulate);
        tcnn::linear_kernel(accumulate_performance_kernel, 0, cudaStreamDefault,
                            n_alive, curr_batch_size, rm_info, _scene_info, curr_rays.results, _subiter_n_steps.data(), _alive_buffer.data(), _network_output_rgbs.data(), _samples.t0.data());

        _ray_dblbuffer.advance();
        subiter++;
    }

    _render_stats.samples_ppx = n_samples_evaluated_total / (float)n_rays;

    _timer.start(TimingState::WriteImage);
    writeImageBuffer_performance(rm_info.cam_info.resolution, _ray_results_final.data(), image_buffer, rm_info.sample_index);
    CUDA_SYNC_CHECK_THROW();

    _timer.syncElapsed();
}