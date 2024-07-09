#pragma once

#include "util/cuda_readwrite_array.h"
#include "renderer_common.h"
#include "common.h"

struct Cache
{
    static constexpr int INIT_ESTIMATED_N_SAMPLES_PER_RAY = 32;
    static constexpr float INIT_ESTIMATED_BRICK_SPARSITY = 0.5f;
    
    static constexpr int BRICK_SIZE = 8;
    static constexpr int BRICK_PADDING = 0;    
    static constexpr int PADDED_BRICK_SIZE = BRICK_SIZE + 2 * BRICK_PADDING;

    static constexpr int2 DATA_ARRAY_DIMS_XY { 1024, 1024 };
    static constexpr int2 DATA_ARRAY_BRICKS_XY { DATA_ARRAY_DIMS_XY.x / PADDED_BRICK_SIZE, DATA_ARRAY_DIMS_XY.y / PADDED_BRICK_SIZE };

    static constexpr int N_DATA_ARRAYS = 2;
    ReadWriteCudaArray3D<float, true> _data_isset_array;
    ReadWriteCudaArray3D<float, true> _data_density_alpha_array;
    ReadWriteCudaArray3D<half4, true> _data_arrays[N_DATA_ARRAYS];

    // Either split up <isset, index> into separate arrays or store in one 3D array as packed index
    // Both are currently set when initializing the cache and can be used in the cache renderers
    char* _brick_isset_array;
    int* _brick_index_array;
    ReadWriteCudaArray3D<int, false> _packed_brick_index_array_3D;
    int3 _brick_array_dims;

    ReadWriteCudaArray2D<float, false> _known_ranges_to;

    int3 _frustum_dims;
    bool _initialized = false;

    // Only used during Cache initialization
    int* _brick_isset_counter;
    char* _brick_isset_array_subiter;

    RaymarchInfo _rm_info;

    void resize(int3 frustum_dims, int estimated_n_bricks);
    void setDataArraysZero(cudaStream_t stream = cudaStreamDefault);


    struct Textures
    {
        cudaTextureObject_t data_isset_tex_pt;
        cudaTextureObject_t data_isset_tex_linear;
        cudaTextureObject_t data_density_alpha_tex_pt;
        cudaTextureObject_t data_density_alpha_tex_linear;
        cudaTextureObject_t data_tex_pt[N_DATA_ARRAYS];
        cudaTextureObject_t data_tex_linear[N_DATA_ARRAYS];
    };

    struct Surfaces
    {
        cudaSurfaceObject_t data_isset_surf;
        cudaSurfaceObject_t data_density_alpha_surf;
        cudaSurfaceObject_t data_surf[N_DATA_ARRAYS];
    };
    Textures textures();
    Surfaces surfaces();

    int3 dataArrayDims() { return _data_isset_array.dims(); }
    int3 dataArrayBricksPerDim() { return _data_isset_array.dims() / PADDED_BRICK_SIZE; }
    int3 brickArrayDims() { return _brick_array_dims; }
    int3 frustumDims() { return _frustum_dims; }
};
