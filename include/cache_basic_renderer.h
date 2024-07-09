
#pragma once

#include "cache_renderer.h"

#include "cache.h"

struct CacheBasicRenderer : public CacheRenderer
{
    CacheBasicRenderer(SceneInfo scene_info, CudaBuffer<uint8_t> &occupancy_grid, NerfNetwork<half> &nerf_network, Cache* cache, Cache* next_cache);

    void render(RaymarchInfo rm_info, RenderBuffer &image_buffer, DebugData* debug_data = nullptr) override;
    void resizeRenderbuffers(int2 resolution) override;

    void postRender(RaymarchInfo rm_info, RenderBuffer &additional_image_buffer);


    // Buffers where payloads_final values are copied to after rendering to allow for cub::ReduceSum
    CudaBuffer<int> _n_samples_contrib_resample, _n_samples_contrib_cache;

    SampleInfoBuffer _samples_resample;
    CudaBuffer<float> _samples_cache_density;
    CudaBuffer<float3> _samples_cache_init_viewdir;

    CudaBuffer<half> _network_output_rgbs_resample;
    CudaBuffer<half> _network_output_rgbs_cache;
    

    struct RenderStats
    {
        int total_n_evaluated_samples_resample;
        int total_n_contributing_samples_resample;

        int total_n_evaluated_samples_cache;
        int total_n_contributing_samples_cache;
    } _render_stats;
};