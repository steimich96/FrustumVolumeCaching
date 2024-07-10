/*
 * Copyright (C) 2024, Michael Steiner, Graz University of Technology.
 * This code is licensed under the MIT license.
 */

#pragma once

#include "common.h"
#include "nerf_network.h"
#include "renderer_common.h"
#include "util/buffer.h"
#include "util/cuda_timer.h"
#include "util/image_buffer.h"

struct SampleInfoBuffer
{
    CudaBuffer<float3> pos;
    CudaBuffer<float3> dir;
    CudaBuffer<float> t0;

    void resize(int n_samples)
    {
        pos.resize(n_samples);
        dir.resize(n_samples);
        t0.resize(n_samples);
    }
};


template<typename T_INIT_INFO, typename T_PAYLOAD, typename T_RESULT>
struct RayBuffer
{
    T_INIT_INFO* init_infos;
    T_PAYLOAD* payloads;
    T_RESULT* results;

    inline RayBuffer<T_INIT_INFO, T_PAYLOAD, T_RESULT> get(int idx)
    {
        return RayBuffer<T_INIT_INFO, T_PAYLOAD, T_RESULT> { init_infos + idx, payloads + idx, results + idx };
    }
};

template<typename T_INIT_INFO, typename T_PAYLOAD, typename T_RESULT>
struct RayDoublebuffer
{
    CudaBuffer<T_INIT_INFO> init_infos;
    CudaBuffer<T_PAYLOAD> payloads;
    CudaBuffer<T_RESULT> results;

    int n_rays;
    int iteration_idx = 0;
    
    RayBuffer<T_INIT_INFO, T_PAYLOAD, T_RESULT> doublebuffer;

    inline void resize(int new_n_rays)
    {
        n_rays = new_n_rays;

        init_infos.resize(n_rays * 2);
        payloads.resize(n_rays * 2);
        results.resize(n_rays * 2);
        doublebuffer = {init_infos.data(), payloads.data(), results.data()};
    }

    void reset() { iteration_idx = 0; }
    void advance() { iteration_idx++; }

    inline RayBuffer<T_INIT_INFO, T_PAYLOAD, T_RESULT> getCurrBuffer() { return doublebuffer.get(((iteration_idx + 1) % 2) * n_rays); }
    inline RayBuffer<T_INIT_INFO, T_PAYLOAD, T_RESULT> getPrevBuffer() { return doublebuffer.get(((iteration_idx + 0) % 2) * n_rays); }
};

struct DebugData
{
    std::vector<RenderBuffer*> buffers;
};

struct Renderer
{
    enum TimingState
    {
        InitRays,
        CompactRays,
        SampleRtx,
        Sample,
        PrepareSamplesResample,
        InferenceResample,
        PrepareSamplesCache,
        InferenceCache,
        Accumulate,
        WriteImage
    };
    static std::map<int, std::string> TIMING_NAMES()
    {
        return std::map<int, std::string> {
            {TimingState::InitRays, "init rays"},
            {TimingState::CompactRays, "compaction"},
            {TimingState::Sample, "sample"},
            {TimingState::PrepareSamplesResample, "prepare samples resample"},
            {TimingState::InferenceResample, "inference resample"},
            {TimingState::PrepareSamplesCache, "prepare samples cache"},
            {TimingState::InferenceCache, "inference cache"},
            {TimingState::Accumulate, "accumulate"},
            {TimingState::WriteImage, "write"}
        };
    };

public:
    Renderer(SceneInfo scene_info, CudaBuffer<uint8_t> &occupancy_grid, NerfNetwork<half> &nerf_network)
        : _scene_info(scene_info), _occupancy_grid(occupancy_grid), _nerf_network(nerf_network), _timer(TIMING_NAMES()) {};

    virtual void render(RaymarchInfo rm_info, RenderBuffer &render_buffer, DebugData* debug_data = nullptr) = 0;
    virtual void resizeRenderbuffers(int2 resolution) = 0;

    SceneInfo _scene_info;
    CudaBuffer<uint8_t>& _occupancy_grid;
    NerfNetwork<half>& _nerf_network;

    CudaTimer<true> _timer;
};

void sample_common(int n_alive, int n_steps_curr_subiter,
                   const RaymarchInfo rm_info, const SceneInfo scene_info, const uint8_t *occupancy_grid,
                   RayPayload *payloads, SampleInfoBuffer &sample_buffer);

int compactPayloads_atomic_common(int n_alive, const RayPayload *prev_payloads,
                           RayPayload *curr_payloads, RayPayload *final_payloads, int *alive_counter);
int compactPayloads_coherent_split(int n_alive, int* new_index_buffer, int *alive_buffer, 
                                   RayBuffer<BaseRayInitInfo, SegmentedRayPayload, BaseRayResult> prev_rays,
                                   RayBuffer<BaseRayInitInfo, SegmentedRayPayload, BaseRayResult> curr_rays,
                                   BaseRayResult *final_results);

void createRayOrder(const int2 resolution, CudaBuffer<int>& ray_order_buffer);
void initRayPayloads_common(const RaymarchInfo rm_info, const SceneInfo scene_info, RayPayload *payloads, const int* ray_order_buffer = nullptr);
void initRayPayloads_split(RaymarchInfo &rm_info, SceneInfo &scene_info,
                           RayBuffer<BaseRayInitInfo, SegmentedRayPayload, BaseRayResult>& ray_buffer, int *alive_buffer, 
                           const int* ray_order_buffer = nullptr);

void writeImageBuffer_common(const int2 resolution, const RayPayload *final_payloads, RenderBuffer &image_buffer, DebugData* debug_data = nullptr);

void accumulate_common(int n_alive, int batch_size, const RaymarchInfo rm_info, const SceneInfo scene_info,
                       RayPayload *payloads, const half *network_output_rgbs, SampleInfoBuffer &sample_buffer);