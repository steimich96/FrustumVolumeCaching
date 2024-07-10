/*
 * Copyright (C) 2024, Michael Steiner, Graz University of Technology.
 * This code is licensed under the MIT license.
 */

#include "my_fully_fused_mlp.h"

#include <mma.h>

// Partially adapted from https://github.com/NVlabs/tiny-cuda-nn/blob/master/src/fully_fused_mlp.cu and other related files

constexpr uint32_t INPUT_SKEW = 8;
constexpr uint32_t WEIGHT_SKEW = 8;

void check_shmem_error(cudaError_t error) {
	if (error != cudaSuccess) {
		throw std::runtime_error{"FullyFusedMLP: insufficient shared memory available on the GPU. Reduce `n_neurons` or use `CutlassMLP` (better compatibility but slower) instead."};
	}
}


template<uint32_t FREQ_ENC_DEGREE>
inline __device__ void freqEncChunk8(const uint32_t offset, const float3 pos, __half2* out)
{
	float x = pos.x;
	float y = pos.y;
	float z = pos.z;
    if (offset==0)
    {
        out[0].x=(__half)__sinf(M_PIf*x);
        out[0].y=(__half)__sinf(2*M_PIf*x);
        out[1].x=(__half)__sinf(4*M_PIf*x);
        out[1].y=(__half)__sinf(8*M_PIf*x);
        out[2].x=(__half)__cosf(M_PIf*x);
        out[2].y=(__half)__cosf(2*M_PIf*x);
        out[3].x=(__half)__cosf(4*M_PIf*x);
        out[3].y=(__half)__cosf(8*M_PIf*x);
    }
    if (offset==8)
    {
        out[0].x=(__half)__sinf(M_PIf*y);
        out[0].y=(__half)__sinf(2*M_PIf*y);
        out[1].x=(__half)__sinf(4*M_PIf*y);
        out[1].y=(__half)__sinf(8*M_PIf*y);
        out[2].x=(__half)__cosf(M_PIf*y);
        out[2].y=(__half)__cosf(2*M_PIf*y);
        out[3].x=(__half)__cosf(4*M_PIf*y);
        out[3].y=(__half)__cosf(8*M_PIf*y);
    }
    if (offset==16)
    {
        out[0].x=(__half)__sinf(M_PIf*z);
        out[0].y=(__half)__sinf(2*M_PIf*z);
        out[1].x=(__half)__sinf(4*M_PIf*z);
        out[1].y=(__half)__sinf(8*M_PIf*z);
        out[2].x=(__half)__cosf(M_PIf*z);
        out[2].y=(__half)__cosf(2*M_PIf*z);
        out[3].x=(__half)__cosf(4*M_PIf*z);
        out[3].y=(__half)__cosf(8*M_PIf*z);
    }
}

template<uint32_t FREQ_ENC_DEGREE>
inline __device__ void freqEncChunk4(const uint32_t offset, const float3 pos, __half2* out)
{
	float x = pos.x;
	float y = pos.y;
	float z = pos.z;
    if (offset==0)
    {
        out[0].x=(__half)__sinf(M_PIf*x);
        out[0].y=(__half)__sinf(2*M_PIf*x);
        out[1].x=(__half)__sinf(4*M_PIf*x);
        out[1].y=(__half)__sinf(8*M_PIf*x);
	}
    if (offset==4)
    {
        out[0].x=(__half)__cosf(M_PIf*x);
        out[0].y=(__half)__cosf(2*M_PIf*x);
        out[1].x=(__half)__cosf(4*M_PIf*x);
        out[1].y=(__half)__cosf(8*M_PIf*x);
    }
    if (offset==8)
    {
        out[0].x=(__half)__sinf(M_PIf*y);
        out[0].y=(__half)__sinf(2*M_PIf*y);
        out[1].x=(__half)__sinf(4*M_PIf*y);
        out[1].y=(__half)__sinf(8*M_PIf*y);
    }
    if (offset==12)
    {
        out[0].x=(__half)__cosf(M_PIf*y);
        out[0].y=(__half)__cosf(2*M_PIf*y);
        out[1].x=(__half)__cosf(4*M_PIf*y);
        out[1].y=(__half)__cosf(8*M_PIf*y);
    }
    if (offset==16)
    {
        out[0].x=(__half)__sinf(M_PIf*z);
        out[0].y=(__half)__sinf(2*M_PIf*z);
        out[1].x=(__half)__sinf(4*M_PIf*z);
        out[1].y=(__half)__sinf(8*M_PIf*z);
    }
    if (offset==20)
    {
        out[0].x=(__half)__cosf(M_PIf*z);
        out[0].y=(__half)__cosf(2*M_PIf*z);
        out[1].x=(__half)__cosf(4*M_PIf*z);
        out[1].y=(__half)__cosf(8*M_PIf*z);
    }
}

inline __device__ void shCoeffChunk8(const uint32_t offset, const float3 dir, __half2* out)
{
	float x = dir.x * 2.f - 1.f;
	float y = dir.y * 2.f - 1.f;
	float z = dir.z * 2.f - 1.f;
	float xy=x*y, xz=x*z, yz=y*z, x2=x*x, y2=y*y, z2=z*z;

	if (offset == 0)
	{
		out[0].x = (__half)(0.28209479177387814f);                          		// 1/(2*sqrt(pi))
		out[0].y = (__half)(-0.48860251190291987f*y);                       		// -sqrt(3)*y/(2*sqrt(pi))
		out[1].x = (__half)(0.48860251190291987f*z);                        		// sqrt(3)*z/(2*sqrt(pi))
		out[1].y = (__half)(-0.48860251190291987f*x);                       		// -sqrt(3)*x/(2*sqrt(pi))
		out[2].x = (__half)(1.0925484305920792f*xy);                                // sqrt(15)*xy/(2*sqrt(pi))
		out[2].y = (__half)(-1.0925484305920792f*yz);                               // -sqrt(15)*yz/(2*sqrt(pi))
		out[3].x = (__half)(0.94617469575755997f*z2 - 0.31539156525251999f);        // sqrt(5)*(3*z2 - 1)/(4*sqrt(pi))
		out[3].y = (__half)(-1.0925484305920792f*xz);                               // -sqrt(15)*xz/(2*sqrt(pi))
	}
	else if (offset == 8)
	{
		out[0].x = (__half)(0.54627421529603959f*x2 - 0.54627421529603959f*y2);     // sqrt(15)*(x2 - y2)/(4*sqrt(pi))
		out[0].y = (__half)(0.59004358992664352f*y*(-3.0f*x2 + y2));                // sqrt(70)*y*(-3*x2 + y2)/(8*sqrt(pi))
		out[1].x = (__half)(2.8906114426405538f*xy*z);                              // sqrt(105)*xy*z/(2*sqrt(pi))
		out[1].y = (__half)(0.45704579946446572f*y*(1.0f - 5.0f*z2));               // sqrt(42)*y*(1 - 5*z2)/(8*sqrt(pi))
		out[2].x = (__half)(0.3731763325901154f*z*(5.0f*z2 - 3.0f));                // sqrt(7)*z*(5*z2 - 3)/(4*sqrt(pi))
		out[2].y = (__half)(0.45704579946446572f*x*(1.0f - 5.0f*z2));               // sqrt(42)*x*(1 - 5*z2)/(8*sqrt(pi))
		out[3].x = (__half)(1.4453057213202769f*z*(x2 - y2));                       // sqrt(105)*z*(x2 - y2)/(4*sqrt(pi))
		out[3].y = (__half)(0.59004358992664352f*x*(-x2 + 3.0f*y2));                // sqrt(70)*x*(-x2 + 3*y2)/(8*sqrt(pi))
	}
}

inline __device__ void shCoeffDiffChunk8(const uint32_t offset, const float3 dir, const float3 other_dir, __half2* out)
{
	__half2 other_coeffs[4];
	shCoeffChunk8(offset, other_dir, other_coeffs);
	shCoeffChunk8(offset, dir, out);

	for (int i = 0; i < 4; i++)
	{
		out[i] = __hsub2(out[i], other_coeffs[i]);
	}	
}

inline __device__ void shCoeffChunk4(const uint32_t offset, const float3 dir, __half2* out)
{
	float x = dir.x * 2.f - 1.f;
	float y = dir.y * 2.f - 1.f;
	float z = dir.z * 2.f - 1.f;
	float xy=x*y, xz=x*z, yz=y*z, x2=x*x, y2=y*y, z2=z*z;

	if (offset == 0)
	{
		out[0].x = (__half)(0.28209479177387814f);                          		// 1/(2*sqrt(pi))
		out[0].y = (__half)(-0.48860251190291987f*y);                       		// -sqrt(3)*y/(2*sqrt(pi))
		out[1].x = (__half)(0.48860251190291987f*z);                        		// sqrt(3)*z/(2*sqrt(pi))
		out[1].y = (__half)(-0.48860251190291987f*x);                       		// -sqrt(3)*x/(2*sqrt(pi))
	}
	else if (offset == 4)
	{
		out[0].x = (__half)(1.0925484305920792f*xy);                                // sqrt(15)*xy/(2*sqrt(pi))
		out[0].y = (__half)(-1.0925484305920792f*yz);                               // -sqrt(15)*yz/(2*sqrt(pi))
		out[1].x = (__half)(0.94617469575755997f*z2 - 0.31539156525251999f);        // sqrt(5)*(3*z2 - 1)/(4*sqrt(pi))
		out[1].y = (__half)(-1.0925484305920792f*xz);                               // -sqrt(15)*xz/(2*sqrt(pi))
	}
	else if (offset == 8)
	{
		out[0].x = (__half)(0.54627421529603959f*x2 - 0.54627421529603959f*y2);     // sqrt(15)*(x2 - y2)/(4*sqrt(pi))
		out[0].y = (__half)(0.59004358992664352f*y*(-3.0f*x2 + y2));                // sqrt(70)*y*(-3*x2 + y2)/(8*sqrt(pi))
		out[1].x = (__half)(2.8906114426405538f*xy*z);                              // sqrt(105)*xy*z/(2*sqrt(pi))
		out[1].y = (__half)(0.45704579946446572f*y*(1.0f - 5.0f*z2));               // sqrt(42)*y*(1 - 5*z2)/(8*sqrt(pi))
	}
	else
	{
		out[0].x = (__half)(0.3731763325901154f*z*(5.0f*z2 - 3.0f));                // sqrt(7)*z*(5*z2 - 3)/(4*sqrt(pi))
		out[0].y = (__half)(0.45704579946446572f*x*(1.0f - 5.0f*z2));               // sqrt(42)*x*(1 - 5*z2)/(8*sqrt(pi))
		out[1].x = (__half)(1.4453057213202769f*z*(x2 - y2));                       // sqrt(105)*z*(x2 - y2)/(4*sqrt(pi))
		out[1].y = (__half)(0.59004358992664352f*x*(-x2 + 3.0f*y2));                // sqrt(70)*x*(-x2 + 3*y2)/(8*sqrt(pi))
	}
}

inline __device__ void shCoeffDiffChunk4(const uint32_t offset, const float3 dir, const float3 other_dir, __half2* out)
{
	__half2 other_coeffs[2];
	shCoeffChunk4(offset, other_dir, other_coeffs);
	shCoeffChunk4(offset, dir, out);

	for (int i = 0; i < 2; i++)
	{
		out[i] = __hsub2(out[i], other_coeffs[i]);
	}	
}


template <typename T>
__device__ void writeShCoeffsToMemory(float3 viewdir, T* out)
{
	// Let compiler figure out how to sequence/reorder these calculations w.r.t. branches
	float x = viewdir.x * 2.f - 1.f;
	float y = viewdir.y * 2.f - 1.f;
	float z = viewdir.z * 2.f - 1.f;

	float xy=x*y, xz=x*z, yz=y*z, x2=x*x, y2=y*y, z2=z*z;

	// SH polynomials generated using scripts/gen_sh.py based on the recurrence relations in appendix A1 of https://www.ppsloan.org/publications/StupidSH36.pdf
	out[0] = (T)(0.28209479177387814f);                          // 1/(2*sqrt(pi))
	out[1] = (T)(-0.48860251190291987f*y);                               // -sqrt(3)*y/(2*sqrt(pi))
	out[2] = (T)(0.48860251190291987f*z);                                // sqrt(3)*z/(2*sqrt(pi))
	out[3] = (T)(-0.48860251190291987f*x);                               // -sqrt(3)*x/(2*sqrt(pi))

	out[4] = (T)(1.0925484305920792f*xy);                                // sqrt(15)*xy/(2*sqrt(pi))
	out[5] = (T)(-1.0925484305920792f*yz);                               // -sqrt(15)*yz/(2*sqrt(pi))
	out[6] = (T)(0.94617469575755997f*z2 - 0.31539156525251999f);                         // sqrt(5)*(3*z2 - 1)/(4*sqrt(pi))
	out[7] = (T)(-1.0925484305920792f*xz);                               // -sqrt(15)*xz/(2*sqrt(pi))
	out[8] = (T)(0.54627421529603959f*x2 - 0.54627421529603959f*y2);                              // sqrt(15)*(x2 - y2)/(4*sqrt(pi))
	out[9] = (T)(0.59004358992664352f*y*(-3.0f*x2 + y2));                         // sqrt(70)*y*(-3*x2 + y2)/(8*sqrt(pi))
	out[10] = (T)(2.8906114426405538f*xy*z);                             // sqrt(105)*xy*z/(2*sqrt(pi))
	out[11] = (T)(0.45704579946446572f*y*(1.0f - 5.0f*z2));                                // sqrt(42)*y*(1 - 5*z2)/(8*sqrt(pi))
	out[12] = (T)(0.3731763325901154f*z*(5.0f*z2 - 3.0f));                         // sqrt(7)*z*(5*z2 - 3)/(4*sqrt(pi))
	out[13] = (T)(0.45704579946446572f*x*(1.0f - 5.0f*z2));                                // sqrt(42)*x*(1 - 5*z2)/(8*sqrt(pi))
	out[14] = (T)(1.4453057213202769f*z*(x2 - y2));                              // sqrt(105)*z*(x2 - y2)/(4*sqrt(pi))
	out[15] = (T)(0.59004358992664352f*x*(-x2 + 3.0f*y2));                                // sqrt(70)*x*(-x2 + 3*y2)/(8*sqrt(pi))
}


template <typename T>
__device__ void writeShCoeffDiffsToMemory(float3 viewdir, float3 other_viewdir, T* out)
{
	// Let compiler figure out how to sequence/reorder these calculations w.r.t. branches
	float x = viewdir.x * 2.0f - 1.0f;
	float y = viewdir.y * 2.0f - 1.0f;
	float z = viewdir.z * 2.0f - 1.0f;

	float ox = other_viewdir.x * 2.0f - 1.0f;
	float oy = other_viewdir.y * 2.0f - 1.0f;
	float oz = other_viewdir.z * 2.0f - 1.0f;

	float xy=x*y, xz=x*z, yz=y*z, x2=x*x, y2=y*y, z2=z*z;
	float oxy=ox*oy, oxz=ox*oz, oyz=oy*oz, ox2=ox*ox, oy2=oy*oy, oz2=oz*oz;

	// SH polynomials generated using scripts/gen_sh.py based on the recurrence relations in appendix A1 of https://www.ppsloan.org/publications/StupidSH36.pdf
	out[0] =  (T)(0.0f);
	out[1] =  (T)((-0.48860251190291987f*y) - (-0.48860251190291987f*oy));
	out[2] =  (T)((0.48860251190291987f*z) - (0.48860251190291987f*oz));
	out[3] =  (T)((-0.48860251190291987f*x) - (-0.48860251190291987f*ox));
	out[4] =  (T)((1.0925484305920792f*xy) - (1.0925484305920792f*oxy));
	out[5] =  (T)((-1.0925484305920792f*yz) - (-1.0925484305920792f*oyz));
	out[6] =  (T)((0.94617469575755997f*z2 - 0.31539156525251999f) - (0.94617469575755997f*oz2 - 0.31539156525251999f));
	out[7] =  (T)((-1.0925484305920792f*xz) - (-1.0925484305920792f*oxz));
	out[8] =  (T)((0.54627421529603959f*x2 - 0.54627421529603959f*y2) - (0.54627421529603959f*ox2 - 0.54627421529603959f*oy2));
	out[9] =  (T)((0.59004358992664352f*y*(-3.0f*x2 + y2)) 	- (0.59004358992664352f*oy*(-3.0f*ox2 + oy2)));
	out[10] = (T)((2.8906114426405538f*xy*z) 				- (2.8906114426405538f*oxy*oz));
	out[11] = (T)((0.45704579946446572f*y*(1.0f - 5.0f*z2)) - (0.45704579946446572f*oy*(1.0f - 5.0f*oz2)));
	out[12] = (T)((0.3731763325901154f*z*(5.0f*z2 - 3.0f)) 	- (0.3731763325901154f*oz*(5.0f*oz2 - 3.0f)));
	out[13] = (T)((0.45704579946446572f*x*(1.0f - 5.0f*z2)) - (0.45704579946446572f*ox*(1.0f - 5.0f*oz2)));
	out[14] = (T)((1.4453057213202769f*z*(x2 - y2)) 		- (1.4453057213202769f*oz*(ox2 - oy2)));
	out[15] = (T)((0.59004358992664352f*x*(-x2 + 3.0f*y2)) 	- (0.59004358992664352f*ox*(-ox2 + 3.0f*oy2)));
}

__device__ __half relu(__half val) {
	return __hmax(val, 0);
}

constexpr uint32_t const_ceil(float num)
{
    return (static_cast<float>(static_cast<uint32_t>(num)) == num)
        ? static_cast<uint32_t>(num)
        : static_cast<uint32_t>(num) + 1;
}

constexpr uint32_t N_SH_COEFFS(uint32_t sh_degree)
{
	switch (sh_degree)
	{
		case 3: return 8;
		case 4: return 16;
		case 5: return 24;
		default: return 16;
	}
}

inline __device__ half2 shfl_reduce_sum_4(half2 val)
{
    half2 result_val = __hadd2(val, __shfl_xor_sync(0xffffffff, val, 1));
    result_val = __hadd2(result_val, __shfl_xor_sync(0xffffffff, result_val, 2));
    return result_val;
}

template <uint32_t WIDTH, uint32_t N_WARPS_PER_BLOCK, uint32_t INPUT_WIDTH, uint32_t N_LATENTS = 8, uint32_t SH_DEGREE_FIRST = 4, uint32_t SH_DEGREE_SECOND = 4, uint32_t OUT_DIMS = 3>
__global__ void kernel_my_mlp_fused_m16n8k8(
	const float* __restrict__ input_viewdir, 
	const float* __restrict__ input_init_viewdir, 
	const __half* __restrict__ input_latents, 
	const __half* __restrict__ weights, 
	__half* __restrict__ out, 
	const uint32_t batch_size)
{
	extern __shared__ __half shmem[];
	__half* act_shmem = shmem;

	constexpr uint32_t WARP_SIZE = 32;
	constexpr uint32_t N_ELEMS_PER_BLOCK = WARP_SIZE * N_WARPS_PER_BLOCK;

	const uint32_t li = threadIdx.x; // lane idx
	const uint32_t wi = threadIdx.y; // warp idx


	const uint32_t is_upper_half_warp = li / 16;
	const uint32_t elem_idx_warp = li;

	const uint32_t shuffle_group_id = li / 8;
	const uint32_t group_id = li / 4;
	const uint32_t col_id = li % 4;

	const uint32_t warp_elem_offset_block = wi * WARP_SIZE;
	const uint32_t warp_elem_offset_global = blockIdx.x * N_ELEMS_PER_BLOCK + warp_elem_offset_block;

	const uint32_t elem_idx_block = warp_elem_offset_block + elem_idx_warp;
	const uint32_t elem_idx_global = warp_elem_offset_global + elem_idx_warp;

	// MMA with a NxK and a KxM matrix for B batches
	constexpr uint32_t B = 2;
	constexpr uint32_t N = 16;
	constexpr uint32_t M = 8;
	constexpr uint32_t K = 8;

	constexpr uint32_t N_INPUT_BLOCKS = INPUT_WIDTH / K;
	constexpr uint32_t N_WEIGHT_BLOCKS = WIDTH / K;

	constexpr uint32_t INPUT_WEIGHT_SKEW = 4;
	constexpr uint32_t PADDED_INPUT_WIDTH = const_ceil(INPUT_WIDTH / 64.f) * 64 + INPUT_WEIGHT_SKEW;
	constexpr uint32_t INPUT_PADDING_WIDTH = PADDED_INPUT_WIDTH - INPUT_WIDTH;

	constexpr uint32_t N_INPUT_WEIGHTS = INPUT_WIDTH * WIDTH;
	constexpr uint32_t N_OUTPUT_WEIGHTS = WIDTH * OUT_DIMS;
	constexpr uint32_t N_WEIGHTS_TOTAL = N_INPUT_WEIGHTS + N_OUTPUT_WEIGHTS;

	constexpr uint32_t INPUT_BUFFER_SIZE_PER_WARP = K * WARP_SIZE;
	constexpr uint32_t INPUT_BUFFER_SHMEM_SIZE = 0; //INPUT_BUFFER_SIZE_PER_WARP * N_WARPS_PER_BLOCK;
	constexpr uint32_t INPUT_WEIGHTS_SHMEM_SIZE = PADDED_INPUT_WIDTH * WIDTH;
	constexpr uint32_t OUTPUT_WEIGHTS_SHMEM_SIZE = N_OUTPUT_WEIGHTS;

	//__half* __restrict__ input_buffer_shmem = act_shmem + (wi * INPUT_BUFFER_SIZE_PER_WARP);
	__half* __restrict__ weights_input_shmem = act_shmem + INPUT_BUFFER_SHMEM_SIZE;
	__half* __restrict__ weights_output_shmem = weights_input_shmem + INPUT_WEIGHTS_SHMEM_SIZE;

	const uint32_t N_HALF_VALUES_LOAD_TYPE = 4; // sizeof(int2) = 4 * sizeof(half)
	const uint32_t N_HALF_VALUES_PER_LOAD = N_ELEMS_PER_BLOCK * N_HALF_VALUES_LOAD_TYPE;

	// Load weights into shared memory	
	TCNN_PRAGMA_UNROLL
	for (uint32_t idx = elem_idx_block * N_HALF_VALUES_LOAD_TYPE; idx < N_INPUT_WEIGHTS; idx += N_HALF_VALUES_PER_LOAD) {
		const uint32_t idx_skewed = idx + (idx / INPUT_WIDTH) * INPUT_PADDING_WIDTH;
		*(int2*)&weights_input_shmem[idx_skewed] = *(int2*)&weights[idx];
	}

	TCNN_PRAGMA_UNROLL
	for (uint32_t idx = elem_idx_block * N_HALF_VALUES_LOAD_TYPE; idx < N_OUTPUT_WEIGHTS; idx += N_HALF_VALUES_PER_LOAD) {
		*(int2*)&weights_output_shmem[idx] = *(int2*)&weights[N_INPUT_WEIGHTS + idx];
	}
	__syncthreads();


	const float3 viewdir = ((float3*) input_viewdir)[elem_idx_global];
	const float3 init_viewdir = ((float3*) input_init_viewdir)[elem_idx_global];

	const uint32_t N_SH_COEFFS_FIRST = N_SH_COEFFS(SH_DEGREE_FIRST);
	const uint32_t N_SH_COEFFS_SECOND = N_SH_COEFFS(SH_DEGREE_SECOND);

	__half results[OUT_DIMS];
	for (int i = 0; i < OUT_DIMS; i++)
		results[i] = 0;

	for (uint32_t weight_block_idx = 0; weight_block_idx < N_WEIGHT_BLOCKS; weight_block_idx++)
	{
		const uint32_t weights_input_offset = weight_block_idx * M * PADDED_INPUT_WIDTH;
		const uint32_t weights_output_block_offset = weight_block_idx * M;

		half2 mma_result_vals[2][2] = {{{0.0f, 0.0f}, {0.0f, 0.0f}}, {{0.0f, 0.0f}, {0.0f, 0.0f}}};

		for (uint32_t input_block_idx = 0; input_block_idx < N_INPUT_BLOCKS; input_block_idx++)
		{
			const uint32_t input_col_idx = input_block_idx * K;

			half2 input_vals[K/2];
			if (input_col_idx < N_LATENTS)
			{
				for (int i = 0; i < (K/2); i++)
				{
					input_vals[i].x = input_latents[elem_idx_global + (input_col_idx + (8 + 2 * i + 0 - shuffle_group_id * 2) % 8) * batch_size];
					input_vals[i].y = input_latents[elem_idx_global + (input_col_idx + (8 + 2 * i + 1 - shuffle_group_id * 2) % 8) * batch_size];
				}
			}
			else if (input_col_idx < N_LATENTS + N_SH_COEFFS_FIRST)
			{
				const uint32_t sh_coeff_offset = input_col_idx - N_LATENTS;

				half2 input_vals_tmp[K/2];
				shCoeffChunk8(sh_coeff_offset, viewdir, input_vals_tmp);

				for (int i = 0; i < (K/2); i++)
				{
					input_vals[(i+shuffle_group_id)%4] = input_vals_tmp[i];
				}
			}
			else if (input_col_idx < N_LATENTS + N_SH_COEFFS_FIRST + N_SH_COEFFS_SECOND)
			{
				const uint32_t sh_coeff_offset = input_col_idx - N_LATENTS - N_SH_COEFFS_FIRST;

				half2 input_vals_tmp[K/2];
				shCoeffDiffChunk8(sh_coeff_offset, viewdir, init_viewdir, input_vals_tmp);

				for (int i = 0; i < (K/2); i++)
				{
					input_vals[(i+shuffle_group_id)%4] = input_vals_tmp[i];
				}
			}
			else
			{
				for (int i = 0; i < (K/2); i++)
				{
					input_vals[i].x = 1.0f;
					input_vals[i].y = 1.0f;
				}
			}

			half2 shuffled_input_vals[K/2];

			for (int i = 0; i < 4; i++)
			{
				half2 tmp = input_vals[i];

				int src_lane = ((4 - (li % 4) + i) % 4) * 8 + li / 4;
				tmp = __shfl_sync(0xffffffff, tmp, src_lane);

				int dst_i = src_lane / 8;	
				if (dst_i == 3) 	 shuffled_input_vals[3] = tmp;
				else if (dst_i == 2) shuffled_input_vals[2] = tmp;
				else if (dst_i == 1) shuffled_input_vals[1] = tmp;
				else 			     shuffled_input_vals[0] = tmp;
			}

			const uint32_t row = (li % 4) * 2;
			const uint32_t col = li / 4;
			half2 weight_input_vals = *(half2*)&weights_input_shmem[weights_input_offset + input_block_idx * K + row + col * PADDED_INPUT_WIDTH];

			for (int it = 0; it < 2; it++)
			{
				unsigned const *A = reinterpret_cast<unsigned const *>(&shuffled_input_vals[it * 2]);
				unsigned const *B = reinterpret_cast<unsigned const *>(&weight_input_vals);
				unsigned const *C = reinterpret_cast<unsigned const *>(&mma_result_vals[it]);
				unsigned *D = reinterpret_cast<unsigned *>(&mma_result_vals[it]);
				asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3}, {%4}, {%5, %6};\n"
					: "=r"(D[0]), "=r"(D[1])
					: "r"(A[0]), "r"(A[1]), 
					"r"(B[0]),
					"r"(C[0]), "r"(C[1]));
			}
		}

		for (int i = 0; i < 2; i++)
			for (int j = 0; j < 2; j++)
				mma_result_vals[i][j] = __hmax2(mma_result_vals[i][j], {0.0f, 0.0f});
		
		
		for (int out_dim_idx = 0; out_dim_idx < OUT_DIMS; out_dim_idx++)
		{
			half2 weight_out_val = *(half2*)&weights_output_shmem[weights_output_block_offset + out_dim_idx * WIDTH + col_id * 2];

			half2 result_tmp[4];
			result_tmp[0] = shfl_reduce_sum_4(__hmul2(mma_result_vals[0][0], weight_out_val));
			result_tmp[1] = shfl_reduce_sum_4(__hmul2(mma_result_vals[0][1], weight_out_val));
			result_tmp[2] = shfl_reduce_sum_4(__hmul2(mma_result_vals[1][0], weight_out_val));
			result_tmp[3] = shfl_reduce_sum_4(__hmul2(mma_result_vals[1][1], weight_out_val));

			half2 result_tmp2;
			if (col_id == 0) 	  result_tmp2 = result_tmp[0];
			else if (col_id == 1) result_tmp2 = result_tmp[1];
			else if (col_id == 2) result_tmp2 = result_tmp[2];
			else          		  result_tmp2 = result_tmp[3];

			results[out_dim_idx] += (result_tmp2.x + result_tmp2.y);
		}
	}

	for (int out_dim_idx = 0; out_dim_idx < OUT_DIMS; out_dim_idx++)
	{
		int src_lane = (li % 8) * 4 + li / 8;
		half tmp_out_val = results[out_dim_idx];
		half out_val = __shfl_sync(0xffffffff, tmp_out_val, src_lane);

		out[elem_idx_global + out_dim_idx * batch_size] = out_val;
	}
}

template <uint32_t WIDTH, uint32_t N_WARPS_PER_BLOCK, uint32_t INPUT_WIDTH, uint32_t N_LATENTS = 8, uint32_t SH_DEGREE_FIRST = 4, uint32_t SH_DEGREE_SECOND = 4, uint32_t OUT_DIMS = 3>
__global__ void kernel_my_mlp_fused_m8n8k4(
	const float* __restrict__ input_viewdir, 
	const float* __restrict__ input_init_viewdir, 
	const __half* __restrict__ input_latents, 
	const __half* __restrict__ weights, 
	__half* __restrict__ out, 
	const uint32_t batch_size)
{
	extern __shared__ __half shmem[];
	__half* act_shmem = shmem;

	constexpr uint32_t WARP_SIZE = 32;
	constexpr uint32_t N_ELEMS_PER_BLOCK = WARP_SIZE * N_WARPS_PER_BLOCK;

	const uint32_t li = threadIdx.x; // lane idx
	const uint32_t wi = threadIdx.y; // warp idx

	// indices according to m8n8k4 MMA fragment layout (https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-fragments-for-mma-m8n8k4-with-f16-floating-point-type)
	const uint32_t is_upper_half_warp = li / 16;
	const uint32_t mma_group_idx_warp = (li - is_upper_half_warp * 16) / 4;
	const uint32_t elem_idx_warp = ((li % 4) + (is_upper_half_warp * 4)) + mma_group_idx_warp * 8;

	const uint32_t warp_elem_offset_block = wi * WARP_SIZE;
	const uint32_t warp_elem_offset_global = blockIdx.x * N_ELEMS_PER_BLOCK + warp_elem_offset_block;

	const uint32_t elem_idx_block = warp_elem_offset_block + elem_idx_warp;
	const uint32_t elem_idx_global = warp_elem_offset_global + elem_idx_warp;

	// MMA with a NxK and a KxM matrix
	constexpr uint32_t N = 8;
	constexpr uint32_t M = 8;
	constexpr uint32_t K = 4;

	constexpr uint32_t N_INPUT_BLOCKS = INPUT_WIDTH / K;
	constexpr uint32_t N_WEIGHT_BLOCKS = WIDTH / 32;

	constexpr uint32_t INPUT_WEIGHT_SKEW = 2;
	constexpr uint32_t PADDED_INPUT_WIDTH = const_ceil(INPUT_WIDTH / 64.f) * 64 + INPUT_WEIGHT_SKEW;
	constexpr uint32_t INPUT_PADDING_WIDTH = PADDED_INPUT_WIDTH - INPUT_WIDTH;
	
	constexpr uint32_t N_INPUT_WEIGHTS = INPUT_WIDTH * WIDTH;
	constexpr uint32_t N_OUTPUT_WEIGHTS = WIDTH * OUT_DIMS;
	constexpr uint32_t N_WEIGHTS_TOTAL = N_INPUT_WEIGHTS + N_OUTPUT_WEIGHTS;

	constexpr uint32_t INPUT_WEIGHTS_SHMEM_SIZE = PADDED_INPUT_WIDTH * WIDTH;
	constexpr uint32_t OUTPUT_WEIGHTS_SHMEM_SIZE = N_OUTPUT_WEIGHTS;

	// Load weights into shared memory
	__half* __restrict__ weights_input_shmem = act_shmem;
	__half* __restrict__ weights_output_shmem = weights_input_shmem + INPUT_WEIGHTS_SHMEM_SIZE;

	const uint32_t N_HALF_VALUES_LOAD_TYPE = 2; // sizeof(half2) = 2 * sizeof(half)
	const uint32_t N_HALF_VALUES_PER_LOAD = N_ELEMS_PER_BLOCK * N_HALF_VALUES_LOAD_TYPE;
	
	TCNN_PRAGMA_UNROLL
	for (uint32_t idx = elem_idx_block * N_HALF_VALUES_LOAD_TYPE; idx < N_INPUT_WEIGHTS; idx += N_HALF_VALUES_PER_LOAD) {
		const uint32_t idx_skewed = idx + (idx / INPUT_WIDTH) * INPUT_PADDING_WIDTH;
		*(half2*)&weights_input_shmem[idx_skewed] = *(half2*)&weights[idx];
	}

	TCNN_PRAGMA_UNROLL
	for (uint32_t idx = elem_idx_block * N_HALF_VALUES_LOAD_TYPE; idx < N_OUTPUT_WEIGHTS; idx += N_HALF_VALUES_PER_LOAD) {
		*(half2*)&weights_output_shmem[idx] = *(half2*)&weights[N_INPUT_WEIGHTS + idx];
	}
	__syncthreads();

	float3 viewdir = ((float3*) input_viewdir)[elem_idx_global];
	float3 init_viewdir = ((float3*) input_init_viewdir)[elem_idx_global];

	uint32_t N_SH_COEFFS_FIRST = N_SH_COEFFS(SH_DEGREE_FIRST);
	uint32_t N_SH_COEFFS_SECOND = N_SH_COEFFS(SH_DEGREE_SECOND);

	__half results[OUT_DIMS];
	for (int i = 0; i < OUT_DIMS; i++)
		results[i] = 0;

	for (uint32_t weight_block_idx = 0; weight_block_idx < N_WEIGHT_BLOCKS; weight_block_idx++)
	{
		uint32_t weights_input_offset = weight_block_idx * WARP_SIZE * PADDED_INPUT_WIDTH;
		uint32_t weights_output_block_offset = weight_block_idx * WARP_SIZE;

		half2 mma_result_vals[4][4] = {
			{{0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}},
			{{0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}},
			{{0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}},
			{{0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}}
		};

		for (uint32_t input_block_idx = 0; input_block_idx < N_INPUT_BLOCKS; input_block_idx++)
		{
			uint32_t input_col_idx = input_block_idx * K;

			half2 input_vals[2];			
			if (input_col_idx < N_LATENTS)
			{
				input_vals[0].x = input_latents[elem_idx_global + (input_col_idx + 0) * batch_size];
				input_vals[0].y = input_latents[elem_idx_global + (input_col_idx + 1) * batch_size];
				input_vals[1].x = input_latents[elem_idx_global + (input_col_idx + 2) * batch_size];
				input_vals[1].y = input_latents[elem_idx_global + (input_col_idx + 3) * batch_size];
			}
			else if (input_col_idx < N_LATENTS + N_SH_COEFFS_FIRST)
			{
				uint32_t sh_coeff_offset = input_col_idx - N_LATENTS;
				shCoeffChunk4(sh_coeff_offset, viewdir, input_vals);
			}
			else if (input_col_idx < N_LATENTS + N_SH_COEFFS_FIRST + N_SH_COEFFS_SECOND)
			{
				uint32_t sh_coeff_offset = input_col_idx - N_LATENTS - N_SH_COEFFS_FIRST;
				shCoeffDiffChunk4(sh_coeff_offset, viewdir, init_viewdir, input_vals);
			}
			else
			{
				for (int i = 0; i < 2; i++)
				{
					input_vals[i].x = 1.0f;
					input_vals[i].y = 1.0f;
				}	
			}

			for (uint32_t weight_sub_block_idx = 0; weight_sub_block_idx < 4; weight_sub_block_idx++)
			{
				uint32_t elem_idx_warp_sub_iter = (elem_idx_warp + weight_sub_block_idx * 8) % WARP_SIZE;

				half2 weight_input_vals[2];
				weight_input_vals[0] = *(half2*)&weights_input_shmem[weights_input_offset + elem_idx_warp_sub_iter * PADDED_INPUT_WIDTH + input_col_idx + 0];
				weight_input_vals[1] = *(half2*)&weights_input_shmem[weights_input_offset + elem_idx_warp_sub_iter * PADDED_INPUT_WIDTH + input_col_idx + 2];

				unsigned const *A = reinterpret_cast<unsigned const *>(&input_vals);
				unsigned const *B = reinterpret_cast<unsigned const *>(&weight_input_vals);
				unsigned const *C = reinterpret_cast<unsigned const *>(&mma_result_vals[weight_sub_block_idx]);
				unsigned *D = reinterpret_cast<unsigned *>(&mma_result_vals[weight_sub_block_idx]);
				asm("mma.sync.aligned.m8n8k4.row.col.f16.f16.f16.f16 {%0, %1, %2, %3}, {%4, %5}, {%6, %7}, {%8, %9, %10, %11};\n"
					: "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
					: "r"(A[0]), "r"(A[1]), 
					"r"(B[0]), "r"(B[1]),
					"r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
			}
		}

		TCNN_PRAGMA_UNROLL
		for (int i = 0; i < 4; i++)
			for (int j = 0; j < 4; j++)
				mma_result_vals[i][j] = __hmax2(mma_result_vals[i][j], {0.0f, 0.0f});

		
		for (int out_dim_idx = 0; out_dim_idx < OUT_DIMS; out_dim_idx++)
		{
			for (uint32_t weight_sub_block_idx = 0; weight_sub_block_idx < 4; weight_sub_block_idx++)
			{
				uint32_t weights_output_sub_block_offset = weights_output_block_offset + out_dim_idx * WIDTH + (((mma_group_idx_warp + weight_sub_block_idx) * M) % 32);

				half2 result_tmp = {0.0f, 0.0f};
				for (int i = 0; i < 4; i++)
				{	
					half2 weight_out_val = *(half2*)&weights_output_shmem[weights_output_sub_block_offset + i * 2];
					result_tmp = __hfma2(mma_result_vals[weight_sub_block_idx][i], weight_out_val, result_tmp);
				}
				results[out_dim_idx] += (result_tmp.x + result_tmp.y);
			}
		}
	}

	for (int out_dim_idx = 0; out_dim_idx < OUT_DIMS; out_dim_idx++)
	{
		out[elem_idx_global + out_dim_idx * batch_size] = results[out_dim_idx];
	}
}


template <uint32_t WIDTH, 
		  uint32_t N_WARPS_PER_BLOCK, 
		  uint32_t INPUT_WIDTH, 
		  uint32_t N_ITERATIONS, 
		  uint32_t N_LATENTS = 8, 
		  uint32_t SH_DEGREE_FIRST = 4, 
		  uint32_t SH_DEGREE_SECOND = 4, 
		  uint32_t FREQ_ENC_DEGREE = 4, 
		  uint32_t OUT_DIMS = 3>
__global__ void kernel_my_mlp_fused_wmma_fast(
	tcnn::MatrixView<const float> input_pos, 
	tcnn::MatrixView<const float> input_viewdir, 
	tcnn::MatrixView<const float> input_init_viewdir, 
	const __half* __restrict__ input_latents, 
	const __half* __restrict__ weights, 
	__half* __restrict__ out, 
	const uint32_t batch_size)
{
	extern __shared__ __half shmem[];
	__half* act_shmem = shmem;

	constexpr uint32_t WARP_SIZE = 32;
	constexpr uint32_t N_ELEMS_PER_ITERATION = WARP_SIZE * N_WARPS_PER_BLOCK;
	constexpr uint32_t N_ELEMS_PER_BLOCK = N_ELEMS_PER_ITERATION * N_ITERATIONS;

	const uint32_t li = threadIdx.x; // lane idx
	const uint32_t wi = threadIdx.y; // warp idx

	const uint32_t warp_elem_offset_block = wi * WARP_SIZE;

	const uint32_t elem_idx_warp = li;
	const uint32_t elem_idx_block = warp_elem_offset_block + elem_idx_warp;
	
	// MMA with a NxK and a KxM matrix for B batches
	constexpr uint32_t B = 2;
	constexpr uint32_t N = 16;
	constexpr uint32_t M = 16;
	constexpr uint32_t K = 16;

	constexpr uint32_t N_INPUT_BLOCKS = INPUT_WIDTH / K;
	constexpr uint32_t N_WEIGHT_BLOCKS = WIDTH / M;

	constexpr uint32_t INPUT_WEIGHT_SKEW = 8;
	constexpr uint32_t PADDED_INPUT_WIDTH = const_ceil(INPUT_WIDTH / 64.f) * 64 + INPUT_WEIGHT_SKEW;
	constexpr uint32_t INPUT_PADDING_WIDTH = PADDED_INPUT_WIDTH - INPUT_WIDTH;
	
	constexpr uint32_t N_INPUT_WEIGHTS = INPUT_WIDTH * WIDTH;
	constexpr uint32_t N_OUTPUT_WEIGHTS = WIDTH * OUT_DIMS;
	constexpr uint32_t N_WEIGHTS_TOTAL = N_INPUT_WEIGHTS + N_OUTPUT_WEIGHTS;

	constexpr uint32_t TMP_BUFFER_SKEW = 8;
	constexpr uint32_t TMP_BUFFER_SIZE_PER_WARP = N * (B * M + TMP_BUFFER_SKEW);
	constexpr uint32_t TMP_BUFFER_SHMEM_SIZE = TMP_BUFFER_SIZE_PER_WARP * N_WARPS_PER_BLOCK;
	constexpr uint32_t INPUT_WEIGHTS_SHMEM_SIZE = PADDED_INPUT_WIDTH * WIDTH;
	constexpr uint32_t OUTPUT_WEIGHTS_SHMEM_SIZE = N_OUTPUT_WEIGHTS;

	// Load weights into shared memory
	__half* __restrict__ tmp_buffer_shmem = act_shmem;
	__half* __restrict__ weights_input_shmem = tmp_buffer_shmem + TMP_BUFFER_SHMEM_SIZE;
	__half* __restrict__ weights_output_shmem = weights_input_shmem + INPUT_WEIGHTS_SHMEM_SIZE;

	const uint32_t N_HALF_VALUES_LOAD_TYPE = 8; // sizeof(int4) = 8 * sizeof(half)
	const uint32_t N_HALF_VALUES_PER_LOAD = N_ELEMS_PER_ITERATION * N_HALF_VALUES_LOAD_TYPE;
	
	TCNN_PRAGMA_UNROLL
	for (uint32_t idx = elem_idx_block * N_HALF_VALUES_LOAD_TYPE; idx < N_INPUT_WEIGHTS; idx += N_HALF_VALUES_PER_LOAD) {
		const uint32_t idx_skewed = idx + (idx / INPUT_WIDTH) * INPUT_PADDING_WIDTH;
		*(int4*)&weights_input_shmem[idx_skewed] = *(int4*)&weights[idx];
	}

	TCNN_PRAGMA_UNROLL
	for (uint32_t idx = elem_idx_block * N_HALF_VALUES_LOAD_TYPE; idx < N_OUTPUT_WEIGHTS; idx += N_HALF_VALUES_PER_LOAD) {
		*(int4*)&weights_output_shmem[idx] = *(int4*)&weights[N_INPUT_WEIGHTS + idx];
	}
	__syncthreads();

	for (int it = 0; it < N_ITERATIONS; it++)
	{
		const uint32_t warp_elem_offset_global = blockIdx.x * N_ELEMS_PER_BLOCK + it * N_ELEMS_PER_ITERATION + warp_elem_offset_block;
		if (warp_elem_offset_global > batch_size)
			break;

		const uint32_t elem_idx_global = warp_elem_offset_global + elem_idx_warp;

		const float3 pos = make_float3(input_pos(0, elem_idx_global), input_pos(1, elem_idx_global), input_pos(2, elem_idx_global));
		const float3 viewdir = make_float3(input_viewdir(0, elem_idx_global), input_viewdir(1, elem_idx_global), input_viewdir(2, elem_idx_global));
		const float3 init_viewdir = make_float3(input_init_viewdir(0, elem_idx_global), input_init_viewdir(1, elem_idx_global), input_init_viewdir(2, elem_idx_global));

		constexpr uint32_t N_SH_COEFFS_FIRST = N_SH_COEFFS(SH_DEGREE_FIRST);
		constexpr uint32_t N_SH_COEFFS_SECOND = N_SH_COEFFS(SH_DEGREE_SECOND);
		constexpr uint32_t N_FREQ_ENC_COEFFS = FREQ_ENC_DEGREE * 3 * 2; // (x, y, z) + (sin, cos)

		const uint32_t warp_tmp_buffer_offset_block = wi * TMP_BUFFER_SIZE_PER_WARP;

		using namespace nvcuda;
		wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::col_major> input_frags[B][N_INPUT_BLOCKS];

		// -------- ENCODING ---------------------------------
		#pragma unroll
		for (uint32_t input_block_idx = 0; input_block_idx < N_INPUT_BLOCKS; input_block_idx++)
		{
			uint32_t input_col_idx = input_block_idx * K;
			uint32_t input_val_offset = 0;

			half2 input_vals[K/2];
			while (input_val_offset < K && input_col_idx < N_LATENTS)
			{
				input_vals[input_val_offset / 2].x = input_latents[elem_idx_global + (input_col_idx + 0) * batch_size];
				input_vals[input_val_offset / 2].y = input_latents[elem_idx_global + (input_col_idx + 1) * batch_size];
				input_col_idx += 2;
				input_val_offset += 2;
			}
			
			while (input_val_offset < K && input_col_idx < (N_LATENTS + N_SH_COEFFS_FIRST))
			{
				const uint32_t sh_coeff_offset = input_col_idx - N_LATENTS;
				shCoeffChunk4(sh_coeff_offset, viewdir, &input_vals[input_val_offset / 2]);
				input_col_idx += 4;
				input_val_offset += 4;
			}
			
			while (input_val_offset < K && input_col_idx < (N_LATENTS + N_SH_COEFFS_FIRST + N_SH_COEFFS_SECOND))
			{
				const uint32_t sh_coeff_offset = input_col_idx - N_LATENTS - N_SH_COEFFS_FIRST;
				shCoeffDiffChunk4(sh_coeff_offset, viewdir, init_viewdir, &input_vals[input_val_offset / 2]);
				input_col_idx += 4;
				input_val_offset += 4;
			}
			
			while (input_val_offset < K && input_col_idx < (N_LATENTS + N_SH_COEFFS_FIRST + N_SH_COEFFS_SECOND + N_FREQ_ENC_COEFFS))
			{
				const uint32_t freq_coeff_offset = input_col_idx - N_LATENTS - N_SH_COEFFS_FIRST - N_SH_COEFFS_SECOND;
				freqEncChunk4<FREQ_ENC_DEGREE>(freq_coeff_offset, pos, &input_vals[input_val_offset / 2]);
				input_col_idx += 4;
				input_val_offset += 4;
			}
			
			while (input_val_offset < K && input_col_idx < INPUT_WIDTH)
			{
				input_vals[input_val_offset / 2] = make_half2(1.0f, 1.0f);
				input_col_idx += 2;
				input_val_offset += 2;				
			}

			for (int i = 0; i < (K/2); i++)
			{
				tmp_buffer_shmem[warp_tmp_buffer_offset_block + elem_idx_warp + (i * 2 + 0) * (B * M + TMP_BUFFER_SKEW)] = input_vals[i].x;
				tmp_buffer_shmem[warp_tmp_buffer_offset_block + elem_idx_warp + (i * 2 + 1) * (B * M + TMP_BUFFER_SKEW)] = input_vals[i].y;
			}

			for (int b = 0; b < B; b++)
				wmma::load_matrix_sync(input_frags[b][input_block_idx], &tmp_buffer_shmem[warp_tmp_buffer_offset_block + b * M], B * M + TMP_BUFFER_SKEW);
		}

		__half results[OUT_DIMS];
		for (int out_dim_idx = 0; out_dim_idx < OUT_DIMS; out_dim_idx++)
			results[out_dim_idx] = 0.0f;

		// -------- MMA of WEIGHT BLOCK: (32xINPUT) x (INPUTx16) -> (32x16) x (16x3) -> (32x3) ---------------------------------
		for (int weight_block_idx = 0; weight_block_idx < N_WEIGHT_BLOCKS; weight_block_idx++)
		{
			uint32_t weights_input_offset = weight_block_idx * M * PADDED_INPUT_WIDTH;
			uint32_t weights_output_block_offset = weight_block_idx * M;

			wmma::fragment<wmma::accumulator, 16, 16, 16, __half> hidden_frags[B];
			for (int b = 0; b < B; b++)
				wmma::fill_fragment(hidden_frags[b], 0.0f);

			// INPUT LAYER: (32xINPUT) x (INPUTx16) -> (32x16)
			for (int in_block_idx = 0; in_block_idx < N_INPUT_BLOCKS; in_block_idx++)
			{
				wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> weights_frag;
				wmma::load_matrix_sync(weights_frag, weights_input_shmem + weights_input_offset + (K * in_block_idx), PADDED_INPUT_WIDTH);

				for (int b = 0; b < B; b++)
					wmma::mma_sync(hidden_frags[b], input_frags[b][in_block_idx], weights_frag, hidden_frags[b]);
			}

			// STORE result of input layer to SHARED MEMORY
			for (int b = 0; b < B; b++)
			{
				// relu is done after loading
				wmma::store_matrix_sync(&tmp_buffer_shmem[warp_tmp_buffer_offset_block + b * M], hidden_frags[b], B * M + TMP_BUFFER_SKEW, wmma::layout_t::mem_col_major);
			}
			
			half2 tmp_result[OUT_DIMS] = {{0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}};

			// LOAD result of input layer from SHARED MEMORY and perform MMA with OUTPUT LAYER weights: (32x16) x (16x3) -> (32x3)
			for (int out_col_idx = 0; out_col_idx < M; out_col_idx += 2)
			{
				half2 hidden_result_vals = make_half2(
					tmp_buffer_shmem[warp_tmp_buffer_offset_block + elem_idx_warp + (out_col_idx + 0) * (B * M + TMP_BUFFER_SKEW)],
					tmp_buffer_shmem[warp_tmp_buffer_offset_block + elem_idx_warp + (out_col_idx + 1) * (B * M + TMP_BUFFER_SKEW)]				
				);
				hidden_result_vals = __hmax2(hidden_result_vals, {0, 0}); // relu

				for (int out_dim_idx = 0; out_dim_idx < OUT_DIMS; out_dim_idx++)
				{
					half2 weight_vals = *(half2*)&weights_output_shmem[weights_output_block_offset + out_col_idx + out_dim_idx * WIDTH];
					tmp_result[out_dim_idx] = __hfma2(hidden_result_vals, weight_vals, tmp_result[out_dim_idx]);
				}
			}

			// ACCUMULATE results of each weight block
			for (int out_dim_idx = 0; out_dim_idx < OUT_DIMS; out_dim_idx++)
				results[out_dim_idx] += (tmp_result[out_dim_idx].x + tmp_result[out_dim_idx].y);
		}

		// STORE accumulated results to global memory
		for (int out_dim_idx = 0; out_dim_idx < OUT_DIMS; out_dim_idx++)
		{
			half2 tmp = make_half2(results[out_dim_idx], __shfl_down_sync(0xFFFFFFFF, results[out_dim_idx], 1));

			if ((elem_idx_global % 2) == 0)
				*(half2*) &out[elem_idx_global + out_dim_idx * batch_size] = tmp;
		}
			
	}
}


template <int WIDTH, int N_WARPS_PER_BLOCK, int INPUT_WIDTH>
__global__ void kernel_my_mlp_fused_wmma(
	const float* __restrict__ input_viewdir, 
	const float* __restrict__ input_init_viewdir, 
	const __half* __restrict__ input_latents, 
	const __half* __restrict__ weights, 
	__half* __restrict__ out, 
	const uint32_t batch_size, 
	const uint32_t n_latents,
    uint32_t sh_degree_viewdir,
    uint32_t sh_degree_init_viewdir)
{
	extern __shared__ __half shmem[];
	__half* act_shmem = shmem;

	// Indices
	const uint32_t li = threadIdx.x; // index in warp ("lane index")
	const uint32_t wi = threadIdx.y; // index in block ("warp index")


	constexpr int N_ITER = 2;
	constexpr int N_INPUT_BLOCKS = INPUT_WIDTH / 16;
	constexpr int N_WEIGHT_BLOCKS = WIDTH / 16;

	constexpr uint32_t N_WEIGHTS_HIDDEN = INPUT_WIDTH * WIDTH;
	constexpr uint32_t N_WEIGHTS_OUTPUT = WIDTH * 16;
	constexpr uint32_t N_WEIGHTS_TOTAL = N_WEIGHTS_HIDDEN + N_WEIGHTS_OUTPUT;

	constexpr uint32_t N_ELEMS_PER_BLOCK = 32 * N_WARPS_PER_BLOCK;

	const uint32_t warp_offset_block = wi * 32;
	const uint32_t block_offset_global = blockIdx.x * N_ELEMS_PER_BLOCK;

	const uint32_t thread_elem_offset_block = (li + warp_offset_block);
	const uint32_t thread_elem_idx_global = thread_elem_offset_block + block_offset_global;
	const uint32_t thread_elem_offset_warp_global = warp_offset_block + block_offset_global;

	constexpr int INPUT_SHMEM_SIZE = (INPUT_WIDTH + INPUT_SKEW) * N_ELEMS_PER_BLOCK;
	constexpr int WEIGHTS_HIDDEN_SHMEM_SIZE = (INPUT_WIDTH + INPUT_SKEW) * WIDTH;

	const uint32_t N_HALF_VALUES_PER_LOAD = N_ELEMS_PER_BLOCK * 8; // loading int4 values -> sizeof(int4) = 8 * sizeof(half)

	__half* __restrict__ input_shmem = act_shmem;
	__half* __restrict__ weights_shmem = act_shmem + INPUT_SHMEM_SIZE;
	__half* __restrict__ weights_output_shmem = weights_shmem + WEIGHTS_HIDDEN_SHMEM_SIZE;

	const uint32_t thread_elem_load_offset = thread_elem_offset_block * 8;
	uint32_t idx = thread_elem_load_offset;

	TCNN_PRAGMA_UNROLL
	for (; idx < N_WEIGHTS_HIDDEN; idx += N_HALF_VALUES_PER_LOAD) {
		const uint32_t idx_skewed = idx + (idx / INPUT_WIDTH) * INPUT_SKEW;
		*(int4*)&weights_shmem[idx_skewed] = *(int4*)&weights[idx];
	}
	__syncthreads();

	TCNN_PRAGMA_UNROLL
	for (; idx < N_WEIGHTS_TOTAL; idx += N_HALF_VALUES_PER_LOAD) {
		if (idx < N_WEIGHTS_HIDDEN)
			continue;

		const uint32_t idx_skewed = (idx - N_WEIGHTS_HIDDEN) + ((idx - N_WEIGHTS_HIDDEN) / WIDTH) * WEIGHT_SKEW;
		*(int4*)&weights_output_shmem[idx_skewed] = *(int4*)&weights[idx];
	}
	__syncthreads();

	__half* input_latents_shmem = input_shmem + (thread_elem_offset_block * (INPUT_WIDTH + INPUT_SKEW));

	TCNN_PRAGMA_UNROLL
	for (int i = 0; i < n_latents; i++)
	{
		input_latents_shmem[i] = input_latents[thread_elem_idx_global + i * batch_size];
	}

	float3 viewdir = ((float3*) input_viewdir)[thread_elem_idx_global];
	float3 init_viewdir = ((float3*) input_init_viewdir)[thread_elem_idx_global];

	const int n_sh_coeffs = 16;
	__half* input_shmem_offset_viewdir_coeffs = input_latents_shmem + n_latents;
	__half* input_shmem_offset_viewdir_diff_coeffs = input_shmem_offset_viewdir_coeffs + n_sh_coeffs;

	writeShCoeffsToMemory(viewdir, input_shmem_offset_viewdir_coeffs);
	writeShCoeffDiffsToMemory(viewdir, init_viewdir, input_shmem_offset_viewdir_diff_coeffs);

	const uint32_t n_actual_inputs = n_latents + 2 * n_sh_coeffs;
	const uint32_t n_padding = INPUT_WIDTH - n_actual_inputs;

	if (n_padding > 0)
	{
		__half* input_shmem_offset_padding = input_shmem_offset_viewdir_diff_coeffs + n_sh_coeffs;
		for (int i = 0; i < n_padding; i++)
		{
			input_shmem_offset_padding[i] = 1.0f;
		}
	}
	__syncwarp();

	using namespace nvcuda;

	wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> input_frags[N_ITER][N_INPUT_BLOCKS];
	for (int i = 0; i < N_ITER; i++)
	{
		for (int in_block_idx = 0; in_block_idx < N_INPUT_BLOCKS; in_block_idx++)
		{
			wmma::load_matrix_sync(input_frags[i][in_block_idx], input_shmem + 16 * in_block_idx + (16 * i + warp_offset_block) * (INPUT_WIDTH + INPUT_SKEW), (INPUT_WIDTH + INPUT_SKEW));
		}
	}

	wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> weights_out_frags[N_WEIGHT_BLOCKS];
	for (int weight_block_idx = 0; weight_block_idx < N_WEIGHT_BLOCKS; weight_block_idx++)
	{
		wmma::load_matrix_sync(weights_out_frags[weight_block_idx], weights_output_shmem + 16 * weight_block_idx, (WIDTH + WEIGHT_SKEW));
	}

	wmma::fragment<wmma::accumulator, 16, 16, 16, __half> acc_out_frags[N_ITER];
	for (int i = 0; i < N_ITER; i++)
	{
		wmma::fill_fragment(acc_out_frags[i], 0.0f);
	}

	for (int weight_block_idx = 0; weight_block_idx < N_WEIGHT_BLOCKS; weight_block_idx++)
	{
		wmma::fragment<wmma::accumulator, 16, 16, 16, __half> acc_hidden_frags[N_ITER];
		for (int i = 0; i < N_ITER; i++)
		{
			wmma::fill_fragment(acc_hidden_frags[i], 0.0f);
		}

		for (int in_block_idx = 0; in_block_idx < N_INPUT_BLOCKS; in_block_idx++)
		{
			wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> weights_frag;
			wmma::load_matrix_sync(weights_frag, weights_shmem + (16 * in_block_idx) + (16 * weight_block_idx) * (INPUT_WIDTH + INPUT_SKEW), INPUT_WIDTH + INPUT_SKEW);

			for (int i = 0; i < N_ITER; i++)
			{
				wmma::mma_sync(acc_hidden_frags[i], input_frags[i][in_block_idx], weights_frag, acc_hidden_frags[i]);
			}
		}

		for (int i = 0; i < N_ITER; i++)
		{
			TCNN_PRAGMA_UNROLL
			for (int t=0; t < acc_hidden_frags[i].num_elements; t++) {
				acc_hidden_frags[i].x[t] = relu(acc_hidden_frags[i].x[t]);
			}

			wmma::store_matrix_sync(input_shmem + 16 * i + (warp_offset_block * (INPUT_WIDTH + INPUT_SKEW)), acc_hidden_frags[i], INPUT_WIDTH + INPUT_SKEW, wmma::layout_t::mem_row_major);
		}

		wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> hidden_frag[N_ITER];
		for (int i = 0; i < N_ITER; i++)
		{
			wmma::load_matrix_sync(hidden_frag[i], input_shmem + 16 * i + (warp_offset_block * (INPUT_WIDTH + INPUT_SKEW)), INPUT_WIDTH + INPUT_SKEW);
			wmma::mma_sync(acc_out_frags[i], hidden_frag[i], weights_out_frags[weight_block_idx], acc_out_frags[i]);
		}
	}

	for (int i = 0; i < N_ITER; i++)
	{
		wmma::store_matrix_sync(out + thread_elem_offset_warp_global + 16 * i, acc_out_frags[i], batch_size, wmma::layout_t::mem_col_major);
	}

}

template <typename T, int WIDTH, int INPUT_WIDTH>
std::enable_if_t<!std::is_same<__half, T>::value> inferenceHeadNetwork_cpp(cudaStream_t stream, 
    tcnn::FullyFusedMLP<T, WIDTH>* m_mlp_head,
    const tcnn::GPUMatrixDynamic<float> &input_pos, 
    const tcnn::GPUMatrixDynamic<float> &input_dir, 
    const tcnn::GPUMatrixDynamic<float> &input_init_viewdir, 
    tcnn::GPUMatrixDynamic<T> &input_latents,
    tcnn::GPUMatrixDynamic<T> &output,
    uint32_t sh_degree_viewdir,
    uint32_t sh_degree_init_viewdir)
{
	throw std::runtime_error{"Our custom fully fused inference pass only supports __half precision."};
}

template <typename T, int WIDTH, int INPUT_WIDTH>
std::enable_if_t<std::is_same<__half, T>::value> inferenceHeadNetwork_cpp(cudaStream_t stream, 
    tcnn::FullyFusedMLP<T, WIDTH>* m_mlp_head,
    const tcnn::GPUMatrixDynamic<float> &input_pos, 
    const tcnn::GPUMatrixDynamic<float> &input_dir, 
    const tcnn::GPUMatrixDynamic<float> &input_init_viewdir, 
    tcnn::GPUMatrixDynamic<T> &input_latents,
    tcnn::GPUMatrixDynamic<T> &output,
    uint32_t sh_degree_viewdir,
    uint32_t sh_degree_init_viewdir)
{
	const uint32_t batch_size = input_latents.cols();
	const uint32_t n_latents = input_latents.rows();

	CHECK_THROW(n_latents % 4 == 0);
	CHECK_THROW(n_latents == 8);

	tcnn::GPUMatrix<T, tcnn::RM>& weights = m_mlp_head->input_weight_matrix(false);

	uint32_t mlp_input_width = m_mlp_head->input_width();

	CHECK_THROW(mlp_input_width % 16 == 0);
	CHECK_THROW(mlp_input_width == INPUT_WIDTH);
	CHECK_THROW(weights.rows() == WIDTH);
	CHECK_THROW(weights.cols() % 16 == 0);
	CHECK_THROW(output.cols() == batch_size);
	CHECK_THROW(input_latents.layout() == tcnn::RM || input_latents.stride() == input_latents.m());

	constexpr uint32_t N_ITERATIONS = 1;
	constexpr uint32_t N_WARPS_PER_BLOCK = 8;
	if (batch_size % (32 * N_WARPS_PER_BLOCK) != 0) {
		throw std::runtime_error{fmt::format("Batch size must be a multiple of {}.", 32 * N_WARPS_PER_BLOCK)};
	}

	const dim3 threads = { 32u, N_WARPS_PER_BLOCK, 1 };

	constexpr uint32_t N_ELEMS_PER_BLOCK = 32 * N_WARPS_PER_BLOCK * N_ITERATIONS;
	uint32_t n_blocks = tcnn::div_round_up(batch_size, N_ELEMS_PER_BLOCK);
	const dim3 blocks = { n_blocks, 1u, 1u };

	constexpr bool USE_WMMA_VARIANT = true;
	if (USE_WMMA_VARIANT)
	{
		constexpr bool USE_FAST_WMMA_VARIANT = true;
		if (USE_FAST_WMMA_VARIANT)
		{
			constexpr uint32_t INPUT_WEIGHT_SKEW = 8;
			constexpr uint32_t PADDED_INPUT_WIDTH = const_ceil(INPUT_WIDTH / 64.f) * 64 + INPUT_WEIGHT_SKEW;

			constexpr uint32_t OUT_DIMS = 3;
			constexpr uint32_t N_INPUT_WEIGHTS = INPUT_WIDTH * WIDTH;
			constexpr uint32_t N_OUTPUT_WEIGHTS = WIDTH * OUT_DIMS;
			constexpr uint32_t N_WEIGHTS_TOTAL = N_INPUT_WEIGHTS + N_OUTPUT_WEIGHTS;

			constexpr uint32_t TMP_BUFFER_SKEW = 8;
			constexpr uint32_t TMP_BUFFER_SIZE_PER_WARP = 16 * (2 * 16 + TMP_BUFFER_SKEW);
			constexpr uint32_t TMP_BUFFER_SHMEM_SIZE = TMP_BUFFER_SIZE_PER_WARP * N_WARPS_PER_BLOCK;
			constexpr uint32_t INPUT_WEIGHTS_SHMEM_SIZE = PADDED_INPUT_WIDTH * WIDTH;
			constexpr uint32_t OUTPUT_WEIGHTS_SHMEM_SIZE = N_OUTPUT_WEIGHTS;

			size_t shmem_size_wmma_fast = sizeof(T) * (TMP_BUFFER_SHMEM_SIZE + INPUT_WEIGHTS_SHMEM_SIZE + OUTPUT_WEIGHTS_SHMEM_SIZE);
			check_shmem_error(cudaFuncSetAttribute(kernel_my_mlp_fused_wmma_fast<WIDTH, N_WARPS_PER_BLOCK, INPUT_WIDTH, N_ITERATIONS>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem_size_wmma_fast));
			kernel_my_mlp_fused_wmma_fast<WIDTH, N_WARPS_PER_BLOCK, INPUT_WIDTH, N_ITERATIONS><<<blocks, threads, shmem_size_wmma_fast, stream>>>(
				input_pos.view(),
				input_dir.view(),
				input_init_viewdir.view(),
				input_latents.data(),
				weights.data(),
				output.data(),
				batch_size
			);
		}
		else
		{
			static_assert(!USE_WMMA_VARIANT || USE_FAST_WMMA_VARIANT || N_ITERATIONS == 1);
			size_t shmem_size_wmma = sizeof(T) * ((INPUT_WIDTH + INPUT_SKEW) * N_ELEMS_PER_BLOCK + (INPUT_WIDTH + INPUT_SKEW) * WIDTH + (WIDTH + WEIGHT_SKEW) * 16);
			check_shmem_error(cudaFuncSetAttribute(kernel_my_mlp_fused_wmma<WIDTH, N_WARPS_PER_BLOCK, INPUT_WIDTH>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem_size_wmma));
			kernel_my_mlp_fused_wmma<WIDTH, N_WARPS_PER_BLOCK, INPUT_WIDTH><<<blocks, threads, shmem_size_wmma, stream>>>(
				input_dir.data(),
				input_init_viewdir.data(),
				input_latents.data(),
				weights.data(),
				output.data(),
				batch_size,
				n_latents,
				sh_degree_viewdir,
				sh_degree_init_viewdir
			);
		}
	}
	else
	{
		static_assert(USE_WMMA_VARIANT || N_ITERATIONS == 1);

		constexpr bool USE_M8N8K4_VARIANT = false;
		if (USE_M8N8K4_VARIANT)
		{
			size_t shmem_size_m8n8k4 = sizeof(T) * ((WIDTH + 2) * WIDTH + WIDTH * 3);
			check_shmem_error(cudaFuncSetAttribute(kernel_my_mlp_fused_m16n8k8<WIDTH, N_WARPS_PER_BLOCK, INPUT_WIDTH>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem_size_m8n8k4));
			kernel_my_mlp_fused_m8n8k4<WIDTH, N_WARPS_PER_BLOCK, INPUT_WIDTH><<<blocks, threads, shmem_size_m8n8k4, stream>>>(
				input_dir.data(),
				input_init_viewdir.data(),
				input_latents.data(),
				weights.data(),
				output.data(),
				batch_size
			);
		}
		else
		{
			size_t shmem_size_m16n8k8 = sizeof(T) * ((WIDTH + 4) * WIDTH + WIDTH * 3);
			check_shmem_error(cudaFuncSetAttribute(kernel_my_mlp_fused_m16n8k8<WIDTH, N_WARPS_PER_BLOCK, INPUT_WIDTH>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem_size_m16n8k8));
			kernel_my_mlp_fused_m16n8k8<WIDTH, N_WARPS_PER_BLOCK, INPUT_WIDTH><<<blocks, threads, shmem_size_m16n8k8, stream>>>(
				input_dir.data(),
				input_init_viewdir.data(),
				input_latents.data(),
				weights.data(),
				output.data(),
				batch_size
			);
		}
	}
}

template <typename T, int WIDTH, int INPUT_WIDTH>
void inferenceHeadNetwork(cudaStream_t stream, 
    tcnn::FullyFusedMLP<T, WIDTH>* m_mlp_head,
    const tcnn::GPUMatrixDynamic<float> &input_pos, 
    const tcnn::GPUMatrixDynamic<float> &input_dir, 
    const tcnn::GPUMatrixDynamic<float> &input_init_viewdir, 
    tcnn::GPUMatrixDynamic<T> &input_latents,
    tcnn::GPUMatrixDynamic<T> &output,
    uint32_t sh_degree_viewdir,
    uint32_t sh_degree_init_viewdir)
{
	inferenceHeadNetwork_cpp<T, WIDTH, INPUT_WIDTH>(stream, 
		m_mlp_head,
		input_pos,
		input_dir, 
		input_init_viewdir, 
		input_latents,
		output,
    	sh_degree_viewdir,
    	sh_degree_init_viewdir);
}


template void inferenceHeadNetwork<__half, 64, 64>(cudaStream_t, 
    tcnn::FullyFusedMLP<__half, 64>*,
    const tcnn::GPUMatrixDynamic<float>&, 
    const tcnn::GPUMatrixDynamic<float>&, 
    const tcnn::GPUMatrixDynamic<float>&, 
    tcnn::GPUMatrixDynamic<__half>&,
    tcnn::GPUMatrixDynamic<__half>&,
    uint32_t,
    uint32_t);

template void inferenceHeadNetwork<__half, 64, 48>(cudaStream_t, 
    tcnn::FullyFusedMLP<__half, 64>*,
    const tcnn::GPUMatrixDynamic<float>&, 
    const tcnn::GPUMatrixDynamic<float>&, 
    const tcnn::GPUMatrixDynamic<float>&, 
    tcnn::GPUMatrixDynamic<__half>&,
    tcnn::GPUMatrixDynamic<__half>&,
    uint32_t,
    uint32_t);


template void inferenceHeadNetwork<__half, 128, 64>(cudaStream_t, 
    tcnn::FullyFusedMLP<__half, 128>*,
    const tcnn::GPUMatrixDynamic<float>&, 
    const tcnn::GPUMatrixDynamic<float>&, 
    const tcnn::GPUMatrixDynamic<float>&, 
    tcnn::GPUMatrixDynamic<__half>&,
    tcnn::GPUMatrixDynamic<__half>&,
    uint32_t,
    uint32_t);

template void inferenceHeadNetwork<__half, 128, 48>(cudaStream_t, 
    tcnn::FullyFusedMLP<__half, 128>*,
    const tcnn::GPUMatrixDynamic<float>&, 
    const tcnn::GPUMatrixDynamic<float>&, 
    const tcnn::GPUMatrixDynamic<float>&, 
    tcnn::GPUMatrixDynamic<__half>&,
    tcnn::GPUMatrixDynamic<__half>&,
    uint32_t,
    uint32_t);



template <typename T, int WIDTH, int INPUT_WIDTH,
		  uint32_t N_LATENTS = 8, 
		  uint32_t SH_DEGREE_FIRST = 4, 
		  uint32_t SH_DEGREE_SECOND = 4, 
		  uint32_t FREQ_ENC_DEGREE = 4>
std::enable_if_t<!std::is_same<__half, T>::value> inferenceHeadNetwork_impl(cudaStream_t stream, 
    tcnn::FullyFusedMLP<T, WIDTH>* m_mlp_head,
    const tcnn::GPUMatrixDynamic<float> &input_pos, 
    const tcnn::GPUMatrixDynamic<float> &input_dir, 
    const tcnn::GPUMatrixDynamic<float> &input_init_viewdir, 
    tcnn::GPUMatrixDynamic<T> &input_latents,
    tcnn::GPUMatrixDynamic<T> &output)
{
	throw std::runtime_error{"Our custom fully fused inference pass only supports __half precision."};
}

template <typename T, int WIDTH,
		  uint32_t N_LATENTS = 8, 
		  uint32_t SH_DEGREE_FIRST = 4, 
		  uint32_t SH_DEGREE_SECOND = 4, 
		  uint32_t FREQ_ENC_DEGREE = 4>
std::enable_if_t<std::is_same<__half, T>::value> inferenceHeadNetwork_impl(cudaStream_t stream, 
    tcnn::FullyFusedMLP<T, WIDTH>* m_mlp_head,
    const tcnn::GPUMatrixDynamic<float> &input_pos, 
    const tcnn::GPUMatrixDynamic<float> &input_dir, 
    const tcnn::GPUMatrixDynamic<float> &input_init_viewdir, 
    tcnn::GPUMatrixDynamic<T> &input_latents,
    tcnn::GPUMatrixDynamic<T> &output,
	bool test)
{
	if (test) return;

	const uint32_t batch_size = input_latents.cols();
	const uint32_t n_latents = input_latents.rows();

	CHECK_THROW(n_latents % 4 == 0);

	tcnn::GPUMatrix<T, tcnn::RM>& weights = m_mlp_head->input_weight_matrix(false);
	uint32_t mlp_input_width = m_mlp_head->input_width();

	constexpr uint32_t N_SH_COEFFS_FIRST = N_SH_COEFFS(SH_DEGREE_FIRST);
	constexpr uint32_t N_SH_COEFFS_SECOND = N_SH_COEFFS(SH_DEGREE_SECOND);
	constexpr uint32_t N_FREQ_ENC_COEFFS = FREQ_ENC_DEGREE * 3 * 2; // (x, y, z) + (sin, cos)
	constexpr uint32_t N_INPUTS = N_LATENTS + N_SH_COEFFS_FIRST + N_SH_COEFFS_SECOND + N_FREQ_ENC_COEFFS;
	constexpr uint32_t INPUT_WIDTH = const_ceil(N_INPUTS / 16.f) * 16;

	CHECK_THROW(mlp_input_width % 16 == 0);
	CHECK_THROW(mlp_input_width == INPUT_WIDTH);
	CHECK_THROW(weights.rows() == WIDTH);
	CHECK_THROW(weights.cols() % 16 == 0);
	CHECK_THROW(output.cols() == batch_size);
	CHECK_THROW(input_latents.layout() == tcnn::RM || input_latents.stride() == input_latents.m());

	constexpr uint32_t N_ITERATIONS = 1;
	constexpr uint32_t N_WARPS_PER_BLOCK = 8;
	if (batch_size % (32 * N_WARPS_PER_BLOCK) != 0) {
		throw std::runtime_error{fmt::format("Batch size must be a multiple of {}.", 32 * N_WARPS_PER_BLOCK)};
	}

	const dim3 threads = { 32u, N_WARPS_PER_BLOCK, 1 };

	constexpr uint32_t N_ELEMS_PER_BLOCK = 32 * N_WARPS_PER_BLOCK * N_ITERATIONS;
	uint32_t n_blocks = tcnn::div_round_up(batch_size, N_ELEMS_PER_BLOCK);
	const dim3 blocks = { n_blocks, 1u, 1u };

	constexpr uint32_t INPUT_WEIGHT_SKEW = 8;
	constexpr uint32_t PADDED_INPUT_WIDTH = const_ceil(INPUT_WIDTH / 64.f) * 64 + INPUT_WEIGHT_SKEW;

	constexpr uint32_t OUT_DIMS = 3;
	constexpr uint32_t N_INPUT_WEIGHTS = INPUT_WIDTH * WIDTH;
	constexpr uint32_t N_OUTPUT_WEIGHTS = WIDTH * OUT_DIMS;
	constexpr uint32_t N_WEIGHTS_TOTAL = N_INPUT_WEIGHTS + N_OUTPUT_WEIGHTS;

	constexpr uint32_t TMP_BUFFER_SKEW = 8;
	constexpr uint32_t TMP_BUFFER_SIZE_PER_WARP = 16 * (2 * 16 + TMP_BUFFER_SKEW);
	constexpr uint32_t TMP_BUFFER_SHMEM_SIZE = TMP_BUFFER_SIZE_PER_WARP * N_WARPS_PER_BLOCK;
	constexpr uint32_t INPUT_WEIGHTS_SHMEM_SIZE = PADDED_INPUT_WIDTH * WIDTH;
	constexpr uint32_t OUTPUT_WEIGHTS_SHMEM_SIZE = N_OUTPUT_WEIGHTS;

	size_t shmem_size_wmma_fast = sizeof(T) * (TMP_BUFFER_SHMEM_SIZE + INPUT_WEIGHTS_SHMEM_SIZE + OUTPUT_WEIGHTS_SHMEM_SIZE);
	check_shmem_error(cudaFuncSetAttribute(kernel_my_mlp_fused_wmma_fast<WIDTH, N_WARPS_PER_BLOCK, INPUT_WIDTH, N_ITERATIONS, N_LATENTS, SH_DEGREE_FIRST, SH_DEGREE_SECOND, FREQ_ENC_DEGREE>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem_size_wmma_fast));
	kernel_my_mlp_fused_wmma_fast<WIDTH, N_WARPS_PER_BLOCK, INPUT_WIDTH, N_ITERATIONS, N_LATENTS, SH_DEGREE_FIRST, SH_DEGREE_SECOND, FREQ_ENC_DEGREE><<<blocks, threads, shmem_size_wmma_fast, stream>>>(
		input_pos.view(),
		input_dir.view(),
		input_init_viewdir.view(),
		input_latents.data(),
		weights.data(),
		output.data(),
		batch_size
	);
}

template <typename T>
void inferenceHeadNetwork_new(cudaStream_t stream, 
    std::unique_ptr<tcnn::Network<T>> &m_mlp_head,
    const tcnn::GPUMatrixDynamic<float> &input_pos, 
    const tcnn::GPUMatrixDynamic<float> &input_dir, 
    const tcnn::GPUMatrixDynamic<float> &input_init_viewdir, 
    tcnn::GPUMatrixDynamic<T> &input_latents,
    tcnn::GPUMatrixDynamic<T> &output,
    uint32_t n_latents,
    uint32_t sh_degree_viewdir,
    uint32_t sh_degree_init_viewdir,
    uint32_t freq_enc_degree,
	bool test)
{
#define CALL_MFFMLP_IMPL(WIDTH, N_LATENTS, SH_DEGREE_FIRST, SH_DEGREE_SECOND, FREQ_ENC_DEGREE) \
	inferenceHeadNetwork_impl<T, WIDTH, N_LATENTS, SH_DEGREE_FIRST, SH_DEGREE_SECOND, FREQ_ENC_DEGREE>(stream, (tcnn::FullyFusedMLP<T, WIDTH>*) m_mlp_head.get(), input_pos, input_dir, input_init_viewdir, input_latents, output, test)

#define CALL_IMPL_FREQ(WIDTH,N_LATENTS, SH_DEGREE_FIRST, SH_DEGREE_SECOND) \
	switch (freq_enc_degree) \
	{ \
	case 4:  { CALL_MFFMLP_IMPL(WIDTH, N_LATENTS, SH_DEGREE_FIRST, SH_DEGREE_SECOND, 4); break; } \
	default: { CALL_MFFMLP_IMPL(WIDTH, N_LATENTS, SH_DEGREE_FIRST, SH_DEGREE_SECOND, 0); break; } \
	} \

#define CALL_IMPL_SH_SECOND(WIDTH,N_LATENTS, SH_DEGREE_FIRST) \
	switch (sh_degree_init_viewdir) \
	{ \
	case 4:  { CALL_IMPL_FREQ(WIDTH, N_LATENTS, SH_DEGREE_FIRST, 4); break; } \
	default: { throw std::invalid_argument{fmt::format("Unsupported SH degree for init viewdir.")}; break; } \
	} \

#define CALL_IMPL_SH_FIRST(WIDTH,N_LATENTS) \
	switch (sh_degree_viewdir) \
	{ \
	case 4:  { CALL_IMPL_SH_SECOND(WIDTH, N_LATENTS, 4); break; } \
	default: { throw std::invalid_argument{fmt::format("Unsupported SH degree for viewdir.")}; break; } \
	} \

#define CALL_IMPL_LATENTS(WIDTH) \
	switch (n_latents) \
	{ \
	case 4:  { CALL_IMPL_SH_FIRST(WIDTH, 4); break; } \
	case 8:  { CALL_IMPL_SH_FIRST(WIDTH, 8); break; } \
	case 16: { CALL_IMPL_SH_FIRST(WIDTH, 16); break; } \
	default: { throw std::invalid_argument{fmt::format("Unsupported Number of latent values.")}; break; } \
	} \

#define CALL_IMPL_WIDTH() \
	switch (m_mlp_head->width(0)) \
	{ \
	case 64:  { CALL_IMPL_LATENTS(64); break; } \
	case 128: { CALL_IMPL_LATENTS(128); break; } \
	default: { throw std::invalid_argument{fmt::format("Unsupported layer WIDTH of the head MLP.")}; break; } \
	} \
	
	CALL_IMPL_WIDTH();

#undef CALL_MFFMLP_IMPL
#undef CALL_IMPL_FREQ
#undef CALL_IMPL_SH_SECOND
#undef CALL_IMPL_SH_FIRST
#undef CALL_IMPL_LATENTS
#undef CALL_IMPL_WIDTH
}

template void inferenceHeadNetwork_new<__half>(
	cudaStream_t, 
    std::unique_ptr<tcnn::Network<__half>> &m_mlp_head,
    const tcnn::GPUMatrixDynamic<float>&, const tcnn::GPUMatrixDynamic<float>&, const tcnn::GPUMatrixDynamic<float>&,
    tcnn::GPUMatrixDynamic<__half>&, tcnn::GPUMatrixDynamic<__half>&,
    uint32_t, uint32_t, uint32_t, uint32_t, bool);