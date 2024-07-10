/*
 * Copyright (C) 2024, Michael Steiner, Graz University of Technology.
 * This code is licensed under the MIT license.
 */

#pragma once

#include "renderer.h"

#ifdef RTX_ENABLED
#include "owl/owl.h"
#include "mesher.h"
#include "programs/init_ray_payloads.h"

extern "C" char init_ray_payloads_ptx[];
#endif


struct PerformanceRenderer : public Renderer
{
public:
    PerformanceRenderer(SceneInfo scene_info, CudaBuffer<uint8_t> &occupancy_grid, NerfNetwork<half> &nerf_network);
    ~PerformanceRenderer() = default;

    void render(RaymarchInfo rm_info, RenderBuffer &render_buffer, DebugData* debug_data = nullptr) override;
    void resizeRenderbuffers(int2 resolution) override;

    RayDoublebuffer<BaseRayInitInfo, SegmentedRayPayload, BaseRayResult> _ray_dblbuffer;
    CudaBuffer<BaseRayResult> _ray_results_final;

    SampleInfoBuffer _samples;

#ifdef RTX_ENABLED
    static constexpr int INIT_RAYS_WITH_RTX = true;
    OWLContext _context;
    
    OWLModule _init_rays_module;
    InitRayPayloadsProgram _init_rays_program;

    OWLGroup _grid_geom;
    TriangleMeshOwl _occupancy_grid_mesh;
#else
    static constexpr int INIT_RAYS_WITH_RTX = false;
#endif

    CudaBuffer<half> _network_intermediate_buffer;
    CudaBuffer<half> _network_output_rgbs;

    CudaBuffer<int> _cub_alive_buffer_tmp;
    CudaBuffer<int> _alive_buffer, _new_index_buffer;
    CudaBuffer<int> _subiter_n_steps;
    CudaBuffer<int> _ray_order_buffer;

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