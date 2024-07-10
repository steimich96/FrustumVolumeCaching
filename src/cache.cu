/*
 * Copyright (C) 2024, Michael Steiner, Graz University of Technology.
 * This code is licensed under the MIT license.
 */

#include "cache.h"

#include "util/cuda_helper.h"

#include "cuda_runtime.h"
#include "raymarch_common.h"

__global__ void setDataArraysZero_kernel(const int3 array_dims, Cache::Surfaces surfaces)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= array_dims.x || y >= array_dims.y || z >= array_dims.z)
        return;

    int3 data_array_idx = make_int3(x, y, z);
    surf3Dwrite<float>(0.0f, surfaces.data_isset_surf, data_array_idx);
    surf3Dwrite<float>(0.0f, surfaces.data_density_alpha_surf, data_array_idx);

    const ushort4 tmp_val = make_ushort4(0U, 0U, 0U, 0U);
    for (int i = 0; i < Cache::N_DATA_ARRAYS; i++)
        surf3Dwrite<ushort4>(tmp_val, surfaces.data_surf[i], data_array_idx);
}

__global__ void initializeBrickArrays_kernel(const int3 brick_array_dims, cudaSurfaceObject_t packed_brick_index_array_3D)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= brick_array_dims.x || y >= brick_array_dims.y || z >= brick_array_dims.z)
        return;

    int3 array_idx = make_int3(x, y, z);
    surf3Dwrite<int>(-1, packed_brick_index_array_3D, array_idx);
}

void Cache::resize(int3 new_frustum_dims, int estimated_n_bricks)
{
    bool dims_changed = new_frustum_dims.x != _frustum_dims.x || new_frustum_dims.y != _frustum_dims.y || new_frustum_dims.z != _frustum_dims.z;
    if (_initialized && dims_changed)
        throw std::runtime_error("Resizing of already allocated cache not implemented yet!\n");

    if (!_initialized)
    {
        std::cout << "Estimated N Bricks: " << estimated_n_bricks << std::endl;
        std::cout << "Frustum Dims: " << new_frustum_dims.x << ", " << new_frustum_dims.y << ", " << new_frustum_dims.z << std::endl;

        _frustum_dims = new_frustum_dims;

        _known_ranges_to.resize(make_int2(_frustum_dims.x, _frustum_dims.y));

        _brick_array_dims = make_int3(divRoundUp(_frustum_dims, BRICK_SIZE));
        int total_brick_array_size = _brick_array_dims.x * _brick_array_dims.y * _brick_array_dims.z;
        _packed_brick_index_array_3D.resize(_brick_array_dims);
        MY_CUDA_CHECK_THROW(cudaMalloc(&_brick_isset_counter, sizeof(int)));
        MY_CUDA_CHECK_THROW(cudaMalloc(&_brick_index_array, total_brick_array_size * sizeof(int)));
        MY_CUDA_CHECK_THROW(cudaMalloc(&_brick_isset_array, total_brick_array_size * sizeof(char)));
        MY_CUDA_CHECK_THROW(cudaMalloc(&_brick_isset_array_subiter, total_brick_array_size * sizeof(char)));

        std::cout << "Frustum Bricks: " << _brick_array_dims.x << ", " << _brick_array_dims.y << ", " << _brick_array_dims.z << std::endl;

        int3 data_array_dims;
        data_array_dims.x = DATA_ARRAY_DIMS_XY.x;
        data_array_dims.y = DATA_ARRAY_DIMS_XY.y;
        data_array_dims.z = divRoundUp(estimated_n_bricks, DATA_ARRAY_BRICKS_XY.x * DATA_ARRAY_BRICKS_XY.y) * PADDED_BRICK_SIZE;

        std::cout << "Data Array Dims: " << data_array_dims.x << ", " << data_array_dims.y << ", " << data_array_dims.z << std::endl;
        std::cout << "Data Array Bricks: " << DATA_ARRAY_BRICKS_XY.x << ", " << DATA_ARRAY_BRICKS_XY.y << ", " << data_array_dims.z / PADDED_BRICK_SIZE << std::endl;

        _data_isset_array.resize(data_array_dims);
        _data_density_alpha_array.resize(data_array_dims);
        for (int i = 0; i < N_DATA_ARRAYS; i++)
            _data_arrays[i].resize(data_array_dims);

        _initialized = true;
    }    
}

void Cache::setDataArraysZero(cudaStream_t stream)
{
    const int BLOCK_SIZE_3D = 8;        
    dim3 block_size3D(BLOCK_SIZE_3D, BLOCK_SIZE_3D, BLOCK_SIZE_3D);

    dim3 data_bricks_grid_size3D = divRoundUp(dataArrayDims(), BLOCK_SIZE_3D);
    setDataArraysZero_kernel<<<data_bricks_grid_size3D, block_size3D, 0, stream>>>(dataArrayDims(), surfaces());

    dim3 bricks_grid_size3D = divRoundUp(brickArrayDims(), BLOCK_SIZE_3D);
    initializeBrickArrays_kernel<<<bricks_grid_size3D, block_size3D, 0, stream>>>(brickArrayDims(), _packed_brick_index_array_3D.surface());

    int total_brick_array_size = _brick_array_dims.x * _brick_array_dims.y * _brick_array_dims.z;
    MY_CUDA_CHECK_THROW(cudaMemset(_brick_isset_array, 0, total_brick_array_size * sizeof(char)));
    MY_CUDA_CHECK_THROW(cudaMemset(_brick_isset_counter, 0, sizeof(int)));
}



Cache::Textures Cache::textures()
{
    Textures textures;
    textures.data_isset_tex_pt = _data_isset_array.texturePt();
    textures.data_isset_tex_linear = _data_isset_array.textureLinear();
    textures.data_density_alpha_tex_pt = _data_density_alpha_array.texturePt();
    textures.data_density_alpha_tex_linear = _data_density_alpha_array.textureLinear();
    for (int i = 0; i < N_DATA_ARRAYS; i++)
    {
        textures.data_tex_pt[i] = _data_arrays[i].texturePt();
        textures.data_tex_linear[i] = _data_arrays[i].textureLinear();
    }
        
    return textures;
}

Cache::Surfaces Cache::surfaces()
{
    Surfaces surfaces;
    surfaces.data_isset_surf = _data_isset_array.surface();
    surfaces.data_density_alpha_surf = _data_density_alpha_array.surface();
    for (int i = 0; i < N_DATA_ARRAYS; i++)
        surfaces.data_surf[i] = _data_arrays[i].surface();
    return surfaces;
}