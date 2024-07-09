
#pragma once

#include "cuda_helper.h"

#include "cub/cub.cuh"

size_t inline __device__ __host__ round_up_pow2(size_t number, size_t multiple)
{
    return (number + multiple - 1) & -multiple;
}


template <typename InputT, typename OutputT>
void inline cubDeviceExclusiveSum(InputT *d_in_array, OutputT *d_out_array, int n_elements)
{
    size_t temp_storage_bytes;
    OutputT *temp_storage = nullptr;
    MY_CUDA_CHECK_THROW(cub::DeviceScan::ExclusiveSum(temp_storage, temp_storage_bytes, d_in_array, d_out_array, n_elements));
    MY_CUDA_CHECK_THROW(cudaMalloc(&temp_storage, temp_storage_bytes));

    MY_CUDA_CHECK_THROW(cub::DeviceScan::ExclusiveSum(temp_storage, temp_storage_bytes, d_in_array, d_out_array, n_elements));
    MY_CUDA_CHECK_THROW(cudaFree(temp_storage));
}

template <typename InputT, typename OutputT>
void inline cubDeviceInclusiveSum(InputT *d_in_array, OutputT *d_out_array, int n_elements)
{
    size_t temp_storage_bytes;
    OutputT *temp_storage = nullptr;
    MY_CUDA_CHECK_THROW(cub::DeviceScan::InclusiveSum(temp_storage, temp_storage_bytes, d_in_array, d_out_array, n_elements));
    MY_CUDA_CHECK_THROW(cudaMalloc(&temp_storage, temp_storage_bytes));

    MY_CUDA_CHECK_THROW(cub::DeviceScan::InclusiveSum(temp_storage, temp_storage_bytes, d_in_array, d_out_array, n_elements));
    MY_CUDA_CHECK_THROW(cudaFree(temp_storage));
}

template <typename InputT, typename OutputT>
OutputT inline cubGetDeviceSum(InputT *d_array, int n_elements, cudaStream_t stream = cudaStreamDefault)
{
    OutputT *d_sum;
    MY_CUDA_CHECK_THROW(cudaMallocAsync(&d_sum, sizeof(OutputT), stream));

    size_t temp_storage_bytes;
    OutputT *temp_storage = nullptr;
    MY_CUDA_CHECK_THROW(cub::DeviceReduce::Sum(temp_storage, temp_storage_bytes, d_array, d_sum, n_elements, stream));
    MY_CUDA_CHECK_THROW(cudaMallocAsync(&temp_storage, temp_storage_bytes, stream));

    OutputT sum;
    MY_CUDA_CHECK_THROW(cub::DeviceReduce::Sum(temp_storage, temp_storage_bytes, d_array, d_sum, n_elements, stream));
    MY_CUDA_CHECK_THROW(cudaMemcpyAsync(&sum, d_sum, sizeof(OutputT), cudaMemcpyDeviceToHost, stream));

    MY_CUDA_CHECK_THROW(cudaFreeAsync(temp_storage, stream));
    MY_CUDA_CHECK_THROW(cudaFreeAsync(d_sum, stream));

    return sum;
}