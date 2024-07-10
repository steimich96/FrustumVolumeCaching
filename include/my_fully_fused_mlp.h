/*
 * Copyright (C) 2024, Michael Steiner, Graz University of Technology.
 * This code is licensed under the MIT license.
 */

#include <tiny-cuda-nn/common.h>

#include <tiny-cuda-nn/networks/fully_fused_mlp.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/network.h>

#include <memory>

template <typename T, int WIDTH=64, int INPUT_WIDTH=64>
void inferenceHeadNetwork(cudaStream_t stream, 
    tcnn::FullyFusedMLP<T, WIDTH>* m_mlp_head,
    const tcnn::GPUMatrixDynamic<float> &input_pos, 
    const tcnn::GPUMatrixDynamic<float> &input_viewdir, 
    const tcnn::GPUMatrixDynamic<float> &input_init_viewdir, 
    tcnn::GPUMatrixDynamic<T> &input_latents,
    tcnn::GPUMatrixDynamic<T> &output,
    uint32_t sh_degree_viewdir,
    uint32_t sh_degree_init_viewdir);

template <typename T>
void inferenceHeadNetwork_new(cudaStream_t stream, 
    std::unique_ptr<tcnn::Network<T>> &m_mlp_head,
    const tcnn::GPUMatrixDynamic<float> &input_pos, 
    const tcnn::GPUMatrixDynamic<float> &input_viewdir, 
    const tcnn::GPUMatrixDynamic<float> &input_init_viewdir,
    tcnn::GPUMatrixDynamic<T> &input_latents,
    tcnn::GPUMatrixDynamic<T> &output,
    uint32_t n_latents,
    uint32_t sh_degree_viewdir,
    uint32_t sh_degree_init_viewdir,
    uint32_t freq_enc_degree, 
    bool test = false);
    