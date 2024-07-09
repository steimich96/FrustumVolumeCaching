
#pragma once

#include <iostream>

#include <tiny-cuda-nn/common.h>

#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/multi_stream.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/network_with_input_encoding.h>

#include "util/cuda_helper.h"
#include "my_fully_fused_mlp.h"

using json = nlohmann::json;

template <typename T>
__global__ void extract_density(
	const uint32_t n_elements,
	const uint32_t density_stride,
	const uint32_t rgbd_stride,
	const T* __restrict__ density,
	T* __restrict__ rgbd
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	rgbd[i * rgbd_stride] = density[i * density_stride];
}

template <typename T>
__global__ void fill(const uint32_t n_elements, T* out, int value) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i >= n_elements) return;
	out[i] = value;
}

template <typename T>
__global__ void sub(const uint32_t num_elements, const T* __restrict__ data_in, T* __restrict__ data_in_out) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;

	data_in_out[i] = data_in[i] - data_in_out[i];
}

template <typename T>
__global__ void frequency_encoding(
	const uint32_t num_elements,
	const uint32_t n_frequencies,
	const tcnn::MatrixView<float> data_in,
	T* data_out)
{
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;

    constexpr int N_DIMS = 3;
    
    float in_data[N_DIMS];
    for (int d = 0; d < N_DIMS; d++)
        in_data[d] = data_in(d, i);
    
    for (uint32_t j = 0; j < (n_frequencies * N_DIMS * 2); j++)
    {
        const uint32_t encoded_input_feature_i = j / (n_frequencies * 2);
        const uint32_t log2_frequency = j % n_frequencies;

        const float phase_shift = ((j / n_frequencies) % 2) * (M_PIf/2);

        const float x = scalbnf(in_data[encoded_input_feature_i], log2_frequency);
        const float input = x * M_PIf + phase_shift;
        data_out[i + j * num_elements] = (T)__sinf(input);
    }
}

template <typename T>
struct NerfNetwork
{
    NerfNetwork() {}

    void init(const json &mlp_base_json, const json &mlp_head_json)
    {
        m_grid_encoding.reset(tcnn::create_encoding<T>(m_n_pos_dims, mlp_base_json["encoding"]));

        m_n_mlp_base_outputs = mlp_base_json["n_output"].get<int>();
        m_n_latents = m_n_mlp_base_outputs;

		json tmp_mlp_base_network_config = mlp_base_json["network"];
		tmp_mlp_base_network_config["n_input_dims"] = m_grid_encoding->padded_output_width();
		tmp_mlp_base_network_config["n_output_dims"] = m_n_mlp_base_outputs;
        m_mlp_base.reset(tcnn::create_network<T>(tmp_mlp_base_network_config));

        uint32_t mlp_head_input_alignment = tcnn::minimum_alignment(mlp_head_json["network"]);
        m_dir_encoding.reset(tcnn::create_encoding<T>(m_n_dir_dims, mlp_head_json["encoding"], mlp_head_input_alignment));
        m_dir_encoding_offset = mlp_head_json.contains("encoding_offset") ? mlp_head_json["encoding_offset"].get<uint32_t>() : 0;
        m_frequency_encoding_degree = mlp_head_json.value<uint32_t>("freq_encoding_degree", 0);
        if (mlp_head_json["encoding"].contains("nested"))
        {
            json dir_encoding_config = mlp_head_json["encoding"]["nested"][0];
            m_dir_encoding_sh_degree = dir_encoding_config["degree"].get<uint32_t>();
        }        

        m_mlp_head_input_width = mlp_head_json.value("n_input", m_dir_encoding->padded_output_width() + m_mlp_base->padded_output_width());
        m_mlp_head_padded_input_width = tcnn::next_multiple(m_mlp_head_input_width, mlp_head_input_alignment);
        // std::cout << "Direnc: " << m_dir_encoding->padded_output_width() << std::endl;

        json tmp_mlp_head_network_config = mlp_head_json["network"];
        tmp_mlp_head_network_config["n_input_dims"] = m_mlp_head_padded_input_width;
        tmp_mlp_head_network_config["n_output_dims"] = 3;
        m_mlp_head.reset(tcnn::create_network<T>(tmp_mlp_head_network_config));
    }

    void init_viewdep(const json &mlp_base_json, const json &mlp_first_head_json, const json &mlp_head_json)
    {
        m_split_head_mlp = true;
        init(mlp_base_json, mlp_head_json);

        uint32_t mlp_first_head_input_alignment = tcnn::minimum_alignment(mlp_first_head_json["network"]);
        m_first_dir_encoding.reset(tcnn::create_encoding<T>(m_n_dir_dims, mlp_first_head_json["encoding"], mlp_first_head_input_alignment));
        m_first_dir_encoding_offset = mlp_first_head_json.contains("encoding_offset") ? mlp_first_head_json["encoding_offset"].get<uint32_t>() : 0;
        if (mlp_first_head_json["encoding"].contains("nested"))
        {
            json first_dir_encoding_config = mlp_first_head_json["encoding"]["nested"][0];
            m_first_dir_encoding_sh_degree = first_dir_encoding_config["degree"].get<uint32_t>();
        }        

        m_first_latent_is_density = mlp_first_head_json.value<bool>("first_latent_is_density", true);
        m_n_latents = mlp_first_head_json["n_output"].get<uint32_t>();
        m_mlp_first_head_padded_input_width = tcnn::next_multiple(mlp_first_head_json["n_input"].get<uint32_t>(), tcnn::minimum_alignment(mlp_first_head_json["network"]));

        json tmp_mlp_first_head_network_config = mlp_first_head_json["network"];
        tmp_mlp_first_head_network_config["n_input_dims"] = m_mlp_first_head_padded_input_width;
        tmp_mlp_first_head_network_config["n_output_dims"] = m_n_latents;
        m_mlp_first_head.reset(tcnn::create_network<T>(tmp_mlp_first_head_network_config));

        try
        {
            tcnn::GPUMatrixDynamic<float> tmp_buffer;
            tcnn::GPUMatrixDynamic<T> tmp_matrix;
            inferenceHeadNetwork_new<T>(cudaStreamDefault, m_mlp_head, tmp_buffer, tmp_buffer, tmp_buffer, tmp_matrix, tmp_matrix, m_n_latents, m_dir_encoding_sh_degree, m_first_dir_encoding_sh_degree, m_frequency_encoding_degree, true);
            m_use_custom_mlp_head = !DISABLE_CUSTOM_NETWORK;

            std::cout << fmt::format("Network uses custom MLP Head implementation: LATENTS = {}, WIDTH = {}, SH1-DEG = {}, SH2-DEG = {}, FREQ-DEG = {}", 
                                     m_n_latents, m_mlp_head->width(0), m_first_dir_encoding_sh_degree, m_dir_encoding_sh_degree, m_frequency_encoding_degree) << std::endl;
        }
        catch(const std::invalid_argument& e)
        {
            m_use_custom_mlp_head = false;
            std::cout << "Cannot use custom MLP Head implementation: " << e.what() << std::endl;
        }
    }

    void inference(cudaStream_t stream, 
        const tcnn::GPUMatrixDynamic<float> &input_pos, 
        const tcnn::GPUMatrixDynamic<float> &input_dir, 
        const tcnn::GPUMatrixDynamic<float> &input_init_viewdir, 
        tcnn::GPUMatrixDynamic<T> &buffer,
        tcnn::GPUMatrixDynamic<T> &output)
    {
        uint32_t batch_size = input_pos.n();

        inferenceLatents(stream, input_pos, input_init_viewdir, buffer);
        inferenceHead(stream, input_pos, input_dir, input_init_viewdir, buffer, output);
    }

    void inferenceLatents(cudaStream_t stream, 
        const tcnn::GPUMatrixDynamic<float> &input_pos, 
        const tcnn::GPUMatrixDynamic<float> &input_init_viewdir, 
        tcnn::GPUMatrixDynamic<T> &buffer)
    {
        uint32_t batch_size = input_pos.n();

		tcnn::GPUMatrixDynamic<T> mlp_base_input = buffer.slice_rows(0, m_grid_encoding->padded_output_width());
        m_grid_encoding->inference_mixed_precision(stream, input_pos, mlp_base_input);

        if (m_split_head_mlp)
        {
            tcnn::GPUMatrixDynamic<T> mlp_first_head_input_matrix = buffer.slice_rows(0, m_mlp_first_head->input_width());
            tcnn::GPUMatrixDynamic<T> mlp_base_output = mlp_first_head_input_matrix.slice_rows(0, m_mlp_base->padded_output_width());
            tcnn::GPUMatrixDynamic<T> first_dir_encoding_output = mlp_first_head_input_matrix.slice_rows(m_mlp_base->padded_output_width(), m_first_dir_encoding->padded_output_width());

            m_mlp_base->inference_mixed_precision(stream, mlp_base_input, mlp_base_output);

            m_first_dir_encoding->inference_mixed_precision(stream, input_init_viewdir, first_dir_encoding_output);

            // offset by 1 because first value is still log-density
		    tcnn::GPUMatrixDynamic<T> mlp_first_head_output = buffer.slice_rows(1, m_mlp_first_head->padded_output_width());
            m_mlp_first_head->inference_mixed_precision(stream, mlp_first_head_input_matrix, mlp_first_head_output);
        }
        else
        {
            tcnn::GPUMatrixDynamic<T> mlp_base_output = buffer.slice_rows(0, m_mlp_base->padded_output_width());            
            m_mlp_base->inference_mixed_precision(stream, mlp_base_input, mlp_base_output);
        }
    }

    void inferenceHead(cudaStream_t stream, 
        const tcnn::GPUMatrixDynamic<float> &input_pos, 
        const tcnn::GPUMatrixDynamic<float> &input_dir, 
        const tcnn::GPUMatrixDynamic<float> &input_init_viewdir, 
        tcnn::GPUMatrixDynamic<T> &mlp_head_input_matrix,
        tcnn::GPUMatrixDynamic<T> &output)
    {
        uint32_t batch_size = input_dir.n();

        tcnn::GPUMatrixDynamic<T> density_row = mlp_head_input_matrix.slice_rows(0, 1);
        tcnn::GPUMatrixDynamic<T> mlp_head_input_matrix_tmp = mlp_head_input_matrix.slice_rows(this->latent_offset(), m_mlp_head->input_width());

        if (m_use_custom_mlp_head)
        {
            tcnn::GPUMatrixDynamic<T> input_latents = mlp_head_input_matrix_tmp.slice_rows(0, m_n_latents);
            inferenceHeadNetwork_new<T>(stream, m_mlp_head, input_pos, input_dir, input_init_viewdir, input_latents, output, m_n_latents, m_dir_encoding_sh_degree, m_first_dir_encoding_sh_degree, m_frequency_encoding_degree);
        }
        else
        {
            tcnn::GPUMatrixDynamic<T> dir_encoding_output = mlp_head_input_matrix_tmp.slice_rows(m_n_latents, m_dir_encoding->padded_output_width());
            m_dir_encoding->inference_mixed_precision(stream, input_dir, dir_encoding_output);

            if (m_split_head_mlp)
            {
                tcnn::GPUMatrixDynamic<T> dir_encoding_diff_buffer = mlp_head_input_matrix_tmp.slice_rows(m_n_latents + m_dir_encoding->padded_output_width(), m_first_dir_encoding->padded_output_width());

                if (input_dir.data() == input_init_viewdir.data())
                {
                    dir_encoding_diff_buffer.memset(0);
                }
                else
                {
                    m_first_dir_encoding->inference_mixed_precision(stream, input_init_viewdir, dir_encoding_diff_buffer);
                    tcnn::linear_kernel(sub<T>, 0, stream, dir_encoding_diff_buffer.n_elements(), dir_encoding_output.data(), dir_encoding_diff_buffer.data());
                }
            }

            if (m_frequency_encoding_degree > 0)
            {
                int freq_enc_offset = m_n_latents + m_dir_encoding->padded_output_width() + (m_split_head_mlp ? m_first_dir_encoding->padded_output_width() : 0);
                tcnn::GPUMatrixDynamic<T> freq_encoding_buffer = mlp_head_input_matrix_tmp.slice_rows(freq_enc_offset, m_frequency_encoding_degree * 3 * 2);
                tcnn::linear_kernel(frequency_encoding<T>, 0, stream, batch_size, m_frequency_encoding_degree, input_pos.view(), freq_encoding_buffer.data());
            }

            if (m_mlp_head_padded_input_width > m_mlp_head_input_width)
            {
                uint32_t padding_width = m_mlp_head_padded_input_width - m_mlp_head_input_width;
                int padding_value = 1; // Strange padding with 1 because of tcnn IdentityEncoding behavior
                tcnn::linear_kernel(fill<T>, 0, stream, batch_size * padding_width, mlp_head_input_matrix_tmp.slice_rows(m_mlp_head_padded_input_width - padding_width, padding_width).data(), padding_value);
            }

            tcnn::GPUMatrixDynamic<T> mlp_head_network_output{ output.data(), m_mlp_head->padded_output_width(), batch_size, output.layout() };
            m_mlp_head->inference_mixed_precision(stream, mlp_head_input_matrix_tmp, mlp_head_network_output);
        }

		tcnn::linear_kernel(extract_density<T>, 0, stream,
			batch_size,
			density_row.layout() == tcnn::AoS ? density_row.stride() : 1,
			output.layout() == tcnn::AoS ? m_mlp_head->padded_output_width() : 1,
			density_row.data(),
			output.data() + 3 * (output.layout() == tcnn::AoS ? 1 : batch_size)
        );
    }

    
    void set_params(T* params_pos_enc, T* params_mlp_base, T* params_mlp_head)
    {
        std::cout << "N Params - Network: " << m_grid_encoding->n_params() << ", " << m_mlp_base->n_params() << ", " << m_mlp_head->n_params() << std::endl;
        m_mlp_base->set_params(params_mlp_base, params_mlp_base, nullptr);
        m_grid_encoding->set_params(params_pos_enc, params_pos_enc, nullptr);

        m_mlp_head->set_params(params_mlp_head, params_mlp_head, nullptr);
        m_dir_encoding->set_params(params_mlp_head + m_mlp_head->n_params(), params_mlp_head + m_mlp_head->n_params(), nullptr);
    }

    void set_params(T* params_mlp_base, T* params_mlp_head)
    {
        set_params(params_mlp_base + m_mlp_base->n_params(), params_mlp_base, params_mlp_head);
    }

    void set_params(T* params)
    {
		size_t offset = 0;

        T* params_mlp_base = params + offset;
        offset += m_mlp_base->n_params();

        T* params_mlp_head = params + offset;
        offset += m_mlp_head->n_params();

        T* params_pos_enc = params + offset;
        offset += m_grid_encoding->n_params();

        T* params_dir_enc = params + offset;
        set_params(params_pos_enc, params_mlp_base, params_mlp_head);
    }
    
    void set_params(T* params_pos_enc, T* params_mlp_base, T* params_mlp_first_head, T* params_mlp_head)
    {
        std::cout << "N Params - Network: " << m_grid_encoding->n_params() << ", " << m_mlp_base->n_params() << ", " << m_mlp_first_head->n_params() << ", " << m_mlp_head->n_params() << std::endl;
        set_params(params_pos_enc, params_mlp_base, params_mlp_head);

        m_mlp_first_head->set_params(params_mlp_first_head, params_mlp_first_head, nullptr);
        m_first_dir_encoding->set_params(params_mlp_first_head + m_mlp_first_head->n_params(), params_mlp_first_head + m_mlp_first_head->n_params(), nullptr);
    }
    
    size_t required_buffer_width()
    {
        uint32_t padded_head_input_width = std::max(m_split_head_mlp ? m_mlp_first_head_padded_input_width : 0, m_use_custom_mlp_head ? m_n_latents : m_mlp_head_padded_input_width);
        return std::max(m_grid_encoding->padded_output_width(), padded_head_input_width) + 1; //(!m_first_latent_is_density);
    }

	size_t n_params() {
		return m_grid_encoding->n_params() + m_mlp_base->n_params() + m_dir_encoding->n_params() + m_mlp_head->n_params() + (m_split_head_mlp ? m_mlp_first_head->n_params() : 0);
	}

    uint32_t padded_output_width()
    {
		return std::max(m_mlp_head->padded_output_width(), (uint32_t) 4);
    }

    uint32_t latent_width()
    {
		return m_n_latents;
    }
    uint32_t padded_latent_width()
    {
		return m_split_head_mlp ? m_mlp_first_head->padded_output_width() : m_mlp_base->padded_output_width();
    }

    uint32_t latent_offset()
    {
        return m_first_latent_is_density ? 0 : 1;
    }


    const bool DISABLE_CUSTOM_NETWORK = false;
    bool m_split_head_mlp = false;
    bool m_first_latent_is_density = true;

    // Hash Encoding and MLP Base
    uint32_t m_n_pos_dims = 3;
    std::unique_ptr<tcnn::Encoding<T>> m_grid_encoding;
    std::unique_ptr<tcnn::Network<T>> m_mlp_base;
    uint32_t m_n_mlp_base_outputs;
    uint32_t m_n_latents;

    // MLP Head
    uint32_t m_use_custom_mlp_head = false;
    uint32_t m_n_dir_dims = 3;
    uint32_t m_frequency_encoding_degree = 0;
    std::unique_ptr<tcnn::Encoding<T>> m_dir_encoding;
    std::unique_ptr<tcnn::Network<T>> m_mlp_head;

    uint32_t m_mlp_head_input_width, m_mlp_head_padded_input_width;
    uint32_t m_dir_encoding_sh_degree = 4;
    uint32_t m_dir_encoding_offset = 0;

    // First MLP Head (for view-dependent first part)
    uint32_t m_mlp_first_head_padded_input_width;
    uint32_t m_first_dir_encoding_sh_degree = 0;
    uint32_t m_first_dir_encoding_offset = 0;

    std::unique_ptr<tcnn::Encoding<T>> m_first_dir_encoding;
    std::unique_ptr<tcnn::Network<T>> m_mlp_first_head;
};