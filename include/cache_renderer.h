/*
 * Copyright (C) 2024, Michael Steiner, Graz University of Technology.
 * This code is licensed under the MIT license.
 */

#pragma once

#include "renderer.h"

#include "cache.h"

#include <array>
#include <mutex>

enum DensityInterpolVariant
{
    Density = 0,                // Use density; Store in own buffer
    Alpha = 1,                  // Use alpha; Store in own buffer
    Intermediates = 2,          // Use first intermediate value (half)
    DensityIntermediates = 3,   // Use density; Store as half in first intermediate value
    AlphaIntermediates = 4      // Use alpha; Store as half in first intermediate value
};

struct CacheSettings
{
    DensityInterpolVariant interpol_variant;
    InterpolFunction interpol_function;
    bool use_hw_interpol;
    bool reweight_intermediates;
    bool disable_reweighting_first_intermediate;
    bool use_inter_brick_interpolation;
};

struct CacheRenderer : public Renderer
{
    CacheRenderer(SceneInfo scene_info, CudaBuffer<uint8_t> &occupancy_grid, NerfNetwork<half> &nerf_network, Cache* cache, Cache* next_cache);
    ~CacheRenderer();

    virtual void render(RaymarchInfo rm_info, RenderBuffer &image_buffer, DebugData* debug_data = nullptr) = 0;
    virtual void resizeRenderbuffers(int2 resolution) = 0;

    void resizeCache(int2 resolution, StepsizeInfo stepsize_info);
    void initCache(RaymarchInfo rm_info, RenderBuffer& debug_image_buffer);
    void initCacheBlockwise(RaymarchInfo rm_info, RenderBuffer& debug_image_buffer);
    
    void swapCache();

    static constexpr int MAX_SUBITER = 1000;
    static constexpr int MIN_N_STEPS_PER_SUBITER = 1;
    static constexpr int MAX_N_STEPS_PER_SUBITER = 8;
 
    static constexpr int TARGET_BATCH_SIZE = 1 << 21;


    static constexpr CacheSettings default_cache_settings
    {
        interpol_variant : DensityInterpolVariant::Density,              // Density Interpolation also works if cache was initialized with different stepsize
        interpol_function : InterpolFunction::Linear,
        use_hw_interpol : true,                                          // Only possible for Linear and Nearest interpolation function
        reweight_intermediates : true,                                   // Reweight, since intermediates are pushed towards zero at borders otherwise
        disable_reweighting_first_intermediate : true,                   // First Intermediate is log(density), which should not be pushed towards zero
        use_inter_brick_interpolation : true                             // If set to false, points are clamped to the brick borders (no inter-brick interpolation)
    };

    CacheSettings cache_settings = default_cache_settings;
    void setCacheSettings(CacheSettings new_cache_settings) { cache_settings = new_cache_settings; }

    Cache* _cache;
    Cache* _cache_next;
    mutable std::mutex _read_mutex;
    std::mutex _write_mutex;
    bool _force_renderbuffer_resize;

    cudaStream_t _low_priority_stream;

    CudaBuffer<BlockPayloadCacheInit> _block_payloads_init;
    CudaBuffer<RayPayloadCacheInit> _ray_payloads_init;

    CudaBuffer<int> _blocks_alive_buffer;
    CudaBuffer<int> _alive_block_idcs;
    CudaBuffer<int> _subiter_block_idcs_compact;
    CudaBuffer<int> _subiter_n_steps_blocks;
    CudaBuffer<int> _subiter_sample_offset_blocks;

    CudaBuffer<int> _cub_alive_cache_buffer_result;
    CudaBuffer<int> _cub_alive_cache_buffer_tmp;
    CudaBuffer<int> _cub_alive_cache_reduce_buffer_tmp;

    SampleInfoBuffer _samples_cache_init;
    CudaBuffer<half> _network_intermediate_buffer_init;



    CudaBuffer<RayPayload> _payloads_doublebuffer;
    CudaBuffer<RayPayload> _payloads_final;

    SampleInfoBuffer _samples_cache;


    CudaBuffer<half> _network_intermediate_buffer;

    CudaBuffer<int> _alive_counter;

    struct InitStats
    {
        float samples_ppx;
        int n_bricks_set;
        int n_bricks_reserved;
    } _init_stats;
};