/*
 * Copyright (C) 2024, Michael Steiner, Graz University of Technology.
 * This code is licensed under the MIT license.
 */

#pragma once

#include "cache_renderer.h"

#ifdef RTX_ENABLED
#include "owl/owl.h"
#include "mesher.h"
#include "programs/init_ray_payloads.h"
#include "programs/sample_segments.h"

extern "C" char init_ray_payloads_ptx[];
extern "C" char sample_segments_ptx[];
#endif


struct CachePerformanceRenderer : public CacheRenderer
{
    enum RenderMode
    {
        BOTH = 0,
        ONLY_CACHE = 1,
        ONLY_RESAMPLED = 2,
    };

    struct __align__(16) SubiterRayEntryCumsum
    {
        int offset_resample;
        int sample_offset_resample;
        int offset_cache;
        int sample_offset_cache;
    };
    struct __align__(16) SubiterEntryCumsumTotal
    {
        int n_rays_resample;
        int n_samples_resample;
        int n_rays_cache;
        int n_samples_cache;
    };
    struct __align__(16) SubiterRayEntry
    {
        int has_resample;
        int n_samples_resample;
        int has_cache;
        int n_samples_cache;

        static inline __host__ __device__ const SubiterRayEntry createFromDiff(CachePerformanceRenderer::SubiterRayEntryCumsum curr, CachePerformanceRenderer::SubiterRayEntryCumsum prev)
        {
            const int tmp_n_samples_resample = curr.sample_offset_resample - prev.sample_offset_resample;
            const int tmp_n_samples_cache = curr.sample_offset_cache - prev.sample_offset_cache;

            return {
                has_resample: tmp_n_samples_resample > 0,
                n_samples_resample: tmp_n_samples_resample,
                has_cache: tmp_n_samples_cache > 0,
                n_samples_cache: tmp_n_samples_cache
            };
        }
    };

    struct SampleCacheEntry
    {
        float t0;
        int packed_brick_index3D;
        float density_interpol;
    };

    CachePerformanceRenderer(SceneInfo scene_info, CudaBuffer<uint8_t> &occupancy_grid, NerfNetwork<half> &nerf_network, Cache* cache, Cache* next_cache);

    void render(RaymarchInfo rm_info, RenderBuffer &render_buffer, DebugData* debug_data = nullptr) override;
    void resizeRenderbuffers(int2 resolution) override;

    void postRender(RaymarchInfo rm_info, RenderBuffer &additional_image_buffer);
    
    CudaBuffer<int> _ray_order_buffer;
    
    RayDoublebuffer<BaseRayInitInfo, SegmentedRayPayload, BaseRayResult> _ray_dblbuffer;
    CudaBuffer<BaseRayResult> _ray_results_final;

    CudaBuffer<float> _samples_resample_t0_tmp;
    CudaBuffer<float> _samples_cache_t0_tmp;
    CudaBuffer<SampleCacheEntry> _samples_cache_tmp;
    CudaBuffer<float3> _samples_cache_froxel_pos;
    CudaBuffer<float3> _samples_cache_init_viewdir;
    
    SampleInfoBuffer _samples_resample;
    CudaBuffer<int> _alive_buffer, _new_index_buffer;
    CudaBuffer<int> _cub_alive_buffer_tmp;
    CudaBuffer<int4> _cub_alive_int4_buffer_tmp;

    CudaBuffer<SubiterRayEntryCumsum> _subiter_ray_entries_cumsum;

    CudaBuffer<int> _subiter_compact_indices_resample, _subiter_compact_indices_cache;
    
#ifdef RTX_ENABLED
    OWLContext _context;
    
    static constexpr int INIT_RAYS_WITH_RTX = true;
    OWLModule _init_rays_module;
    InitRayPayloadsProgram _init_rays_program;
    
    static constexpr int SAMPLE_WITH_RTX = false;
    OWLModule _sample_module;
    SampleSegmentsProgram _sample_program;

    OWLGroup _grid_geom;
    TriangleMeshOwl _occupancy_grid_mesh;
#else
    static constexpr int INIT_RAYS_WITH_RTX = false;
    static constexpr int SAMPLE_WITH_RTX = false;
#endif

    CudaBuffer<half> _network_output_resample_rgbs;
    CudaBuffer<half> _network_output_cache_rgbs;
    

    RenderMode _render_mode = RenderMode::BOTH;

    struct RenderStats
    {
        float samples_ppx = 0;
        float resample_samples_evaluated = 0.f;
        float cache_samples_evaluated = 0.f;
        uint64_t n_resamples = 0;
        uint64_t n_cache_hits = 0;
    };
    RenderStats _render_stats;
};