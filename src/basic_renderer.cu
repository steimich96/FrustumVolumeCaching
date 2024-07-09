
#include "basic_renderer.h"

#include "raymarch_common.h"
#include "util/cub_helper.h"

BasicRenderer::BasicRenderer(SceneInfo scene_info, CudaBuffer<uint8_t> &occupancy_grid, NerfNetwork<half> &nerf_network)
    : Renderer(scene_info, occupancy_grid, nerf_network)
{
}

void BasicRenderer::resizeRenderbuffers(int2 resolution)
{
    int n_rays = resolution.x * resolution.y;
    
    _payloads_doublebuffer.resize(n_rays * 2);
    _payloads_final.resize(n_rays);

    int buffer_size = tcnn::next_multiple(max(n_rays, TARGET_BATCH_SIZE) * MIN_N_STEPS_PER_SUBITER, (int)tcnn::batch_size_granularity);

    _samples.resize(buffer_size);
    _network_intermediate_buffer.resize(buffer_size * _nerf_network.required_buffer_width());
    _network_output_rgbs.resize(buffer_size * _nerf_network.padded_output_width());

    _alive_counter.resize(1);
}

void BasicRenderer::render(RaymarchInfo rm_info, RenderBuffer &image_buffer, DebugData* debug_data)
{
    int n_rays = rm_info.cam_info.resolution.x * rm_info.cam_info.resolution.y;
    int n_alive = n_rays;

    constexpr int BLOCK_SIZE_1D = 128;
    dim3 grid_size_alive_1D(divRoundUp(n_alive, BLOCK_SIZE_1D));

    initRayPayloads_common(rm_info, _scene_info, _payloads_doublebuffer.data());
    //CUDA_SYNC_CHECK_THROW();

    int n_samples_evaluated_total = 0;

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
        n_samples_evaluated_total += curr_batch_size;

        sample_common(n_alive, n_steps_curr_subiter, rm_info, _scene_info, _occupancy_grid.data(), curr_payloads, _samples);

        tcnn::GPUMatrix<float> network_input_pos((float *)_samples.pos.data(), 3, curr_batch_size);
        tcnn::GPUMatrix<float> network_input_dir((float *)_samples.dir.data(), 3, curr_batch_size);

        tcnn::GPUMatrix<half, tcnn::RM> mlp_head_input_matrix(_network_intermediate_buffer.data(), _nerf_network.m_mlp_head->input_width(), curr_batch_size);
        tcnn::GPUMatrix<half, tcnn::RM> rgbsigma_matrix(_network_output_rgbs.data(), _nerf_network.padded_output_width(), curr_batch_size);

        _nerf_network.inference(cudaStreamDefault, network_input_pos, network_input_dir, network_input_dir, mlp_head_input_matrix, rgbsigma_matrix);

        accumulate_common(n_alive, curr_batch_size, rm_info, _scene_info, curr_payloads, _network_output_rgbs.data(), _samples);

        subiter++;
    }

    _render_stats.samples_ppx = n_samples_evaluated_total / (float) n_rays;

    writeImageBuffer_common(rm_info.cam_info.resolution, _payloads_final.data(), image_buffer);
    // CUDA_SYNC_CHECK_THROW();
}