/*
 * Copyright (C) 2024, Michael Steiner, Graz University of Technology.
 * This code is licensed under the MIT license.
 */

#pragma once

#include "cuda_helper.h"
#include "helper_math_extension.h"

#include <memory>

template <typename T> inline cudaChannelFormatDesc createChannelDesc() { return cudaCreateChannelDesc<T>(); }

template <> inline cudaChannelFormatDesc createChannelDesc<half4>() { return cudaCreateChannelDescHalf4(); }
template <> inline cudaChannelFormatDesc createChannelDesc<half2>() { return cudaCreateChannelDescHalf2(); }
template <> inline cudaChannelFormatDesc createChannelDesc<half>() { return cudaCreateChannelDescHalf(); }

template <typename T, bool LINEAR_FILTER_ENABLED>
class ReadWriteCudaArray
{
public:
    void free()
    {
        MY_CUDA_CHECK_THROW(cudaDestroyTextureObject(_texture_pt));
        MY_CUDA_CHECK_THROW(cudaDestroyTextureObject(_texture_linear));
        MY_CUDA_CHECK_THROW(cudaDestroySurfaceObject(_surface));
        MY_CUDA_CHECK_THROW(cudaFreeArray(_array));        
    }

    cudaSurfaceObject_t surface() { return _surface; }
    cudaTextureObject_t texturePt() { return _texture_pt; }
    cudaTextureObject_t textureLinear() { 
        if (LINEAR_FILTER_ENABLED)
            return _texture_linear;
        else
            return _texture_pt;
    }

protected:
    cudaTextureObject_t createTexture(cudaTextureFilterMode filter_mode, cudaTextureAddressMode address_mode = cudaAddressModeBorder)
    {
        cudaResourceDesc tex_res_desc;
        memset(&tex_res_desc, 0, sizeof(cudaResourceDesc));
        tex_res_desc.resType = cudaResourceTypeArray;
        tex_res_desc.res.array.array = _array;

        cudaTextureDesc tex_desc;
        memset(&tex_desc, 0, sizeof(cudaTextureDesc));

        tex_desc.normalizedCoords = false;
        tex_desc.filterMode = filter_mode;
        tex_desc.addressMode[0] = address_mode;
        tex_desc.addressMode[1] = address_mode;
        tex_desc.addressMode[2] = address_mode;
        tex_desc.readMode = cudaReadModeElementType;

        cudaTextureObject_t texture;
        MY_CUDA_CHECK_THROW(cudaCreateTextureObject(&texture, &tex_res_desc, &tex_desc, NULL));

        return texture;
    }

    cudaSurfaceObject_t createSurface()
    {
        cudaResourceDesc surf_res_desc;
        memset(&surf_res_desc, 0, sizeof(cudaResourceDesc));
        surf_res_desc.resType = cudaResourceTypeArray;
        surf_res_desc.res.array.array = _array;

        cudaSurfaceObject_t surface;
        MY_CUDA_CHECK_THROW(cudaCreateSurfaceObject(&surface, &surf_res_desc));

        return surface;
    }

    void init(cudaTextureAddressMode address_mode = cudaAddressModeBorder)
    {
        _surface = createSurface();
        _texture_pt = createTexture(cudaFilterModePoint, address_mode);

        if (LINEAR_FILTER_ENABLED)
            _texture_linear = createTexture(cudaFilterModeLinear, address_mode);
    }

    cudaArray_t _array;

    cudaSurfaceObject_t _surface;
    cudaTextureObject_t _texture_pt;
    cudaTextureObject_t _texture_linear;
};

template <typename T, bool LINEAR_FILTER_ENABLED>
class ReadWriteCudaArray3D : public ReadWriteCudaArray<T, LINEAR_FILTER_ENABLED>
{
public:
    void resize(int3 dims, cudaTextureAddressMode address_mode = cudaAddressModeBorder)
    {
        _dims = dims;

        cudaChannelFormatDesc desc = createChannelDesc<T>();
        MY_CUDA_CHECK_THROW(cudaMalloc3DArray(&(this->_array), &desc, make_cudaExtent(dims.x, dims.y, dims.z), cudaArraySurfaceLoadStore));

        this->init(address_mode);
    }

    void memset(int value)
    {
        T *dData = NULL;
        cudaMalloc((void **) &dData, size()*sizeof(T));
        cudaMemset(dData, 0, size()*sizeof(T));
        cudaMemcpy3DParms p = {0};
        p.srcPtr = make_cudaPitchedPtr(dData, _dims.x*sizeof(T), _dims.x, _dims.y);
        p.srcPos = make_cudaPos(0,0,0);
        p.dstArray = this->_array;
        p.dstPos = make_cudaPos(0,0,0);
        p.extent = make_cudaExtent(_dims.x, _dims.y, 1);
        p.kind   = cudaMemcpyDefault;
        for (int i = 0; i < _dims.z; i++)
        {
            cudaMemcpy3D(&p);
            p.dstPos = make_cudaPos(0,0, i+1);
        }
        cudaFree(dData);
    }

    int3 dims() { return _dims; }
    int size() { return _dims.x * _dims.y * _dims.z; }

private:
    int3 _dims;
};

template <typename T, bool LINEAR_FILTER_ENABLED>
class ReadWriteCudaArray2D : public ReadWriteCudaArray<T, LINEAR_FILTER_ENABLED>
{
public:
    void resize(int2 dims, cudaTextureAddressMode address_mode = cudaAddressModeBorder)
    {
        _dims = dims;

        cudaChannelFormatDesc desc = cudaCreateChannelDesc<T>();
        MY_CUDA_CHECK_THROW(cudaMallocArray(&(this->_array), &desc, dims.x, dims.y, cudaArraySurfaceLoadStore));

        this->init(address_mode);
    }

    void download(T* h_buffer)
    {
        MY_CUDA_CHECK_THROW(cudaMemcpy2DFromArray(h_buffer, _dims.x * sizeof(T), this->_array, 0, 0, _dims.x * sizeof(T), _dims.y, cudaMemcpyDeviceToHost));
    }

    void upload(T* h_buffer)
    {
        MY_CUDA_CHECK_THROW(cudaMemcpy2DToArray(this->_array, 0, 0, h_buffer, _dims.x * sizeof(T), _dims.x * sizeof(T), _dims.y, cudaMemcpyHostToDevice));
    }

    int2 dims() { return _dims; }
    int size() { return _dims.x * _dims.y; }

private:
    int2 _dims;
};