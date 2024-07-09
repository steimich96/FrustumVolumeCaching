
#pragma once

#include "renderer.h"

struct BasicRenderer : public Renderer
{
public:
    BasicRenderer(SceneInfo scene_info, CudaBuffer<uint8_t> &occupancy_grid, NerfNetwork<half> &nerf_network);
    ~BasicRenderer() = default;

    void render(RaymarchInfo rm_info, RenderBuffer &image_buffer, DebugData* debug_data = nullptr) override;
    void resizeRenderbuffers(int2 resolution) override;


    CudaBuffer<RayPayload> _payloads_doublebuffer;
    CudaBuffer<RayPayload> _payloads_final;

    SampleInfoBuffer _samples;

    CudaBuffer<half> _network_intermediate_buffer;
    CudaBuffer<half> _network_output_rgbs;

    CudaBuffer<int> _alive_counter;

    static constexpr int MAX_SUBITER = 1000;
    static constexpr int MIN_N_STEPS_PER_SUBITER = 1;
    static constexpr int MAX_N_STEPS_PER_SUBITER = 8;

    static constexpr int TARGET_BATCH_SIZE = 1 << 21;

    struct RenderStats
    {
        float samples_ppx;
    };
    RenderStats _render_stats;
};