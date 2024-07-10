/*
 * Copyright (C) 2024, Michael Steiner, Graz University of Technology.
 * This code is licensed under the MIT license.
 */

#pragma once

#include <cuda_runtime.h>

#include "cuda_helper.h"

#include <vector>
#include <fstream>
#include <filesystem>

#include "bmpwriter.h"

template <typename T>
struct CudaBuffer
{
    using type = T;

    CudaBuffer() {};
    CudaBuffer(int n_elements)
    {
        resize(n_elements);
    }
    CudaBuffer(int n_elements, int init_val) : CudaBuffer(n_elements)
    {
        memset(init_val);
    }

    CudaBuffer (const CudaBuffer&) = delete;
    CudaBuffer& operator= (const CudaBuffer&) = delete;

    ~CudaBuffer()
    {
        if (d_ptr) free();
    }

    void resize(size_t new_n_elements)
    {
        if (d_ptr)
        {
            if (new_n_elements != this->n_elements)
                free();
            else
                return;
        }
        alloc(new_n_elements);
    }

    void upload(std::vector<T> h_vec)
    {
        resize(h_vec.size());
        MY_CUDA_CHECK_THROW(cudaMemcpy(d_ptr, h_vec.data(), sizeof(T) * h_vec.size(), cudaMemcpyHostToDevice));
    }

    void upload(T* h_ptr, size_t n_elements)
    {
        resize(n_elements);
        MY_CUDA_CHECK_THROW(cudaMemcpy(d_ptr, h_ptr, sizeof(T) * n_elements, cudaMemcpyHostToDevice));
    }

    void download(std::vector<T>& h_vec)
    {
        h_vec.resize(n_elements);
        MY_CUDA_CHECK_THROW(cudaMemcpy(h_vec.data(), d_ptr, sizeof(T) * n_elements, cudaMemcpyDeviceToHost));
    }

    T downloadFirst()
    {
        T val;
        MY_CUDA_CHECK_THROW(cudaMemcpy(&val, d_ptr, sizeof(T), cudaMemcpyDeviceToHost));
        return val;
    }

    void free()
    {
        MY_CUDA_CHECK_THROW(cudaFree(d_ptr));
        d_ptr = nullptr;
        n_elements = 0;
    }

    void alloc(size_t new_n_elements)
    {
        MY_CUDA_CHECK_THROW(cudaMalloc(&d_ptr, sizeof(T) * new_n_elements));
        n_elements = new_n_elements;
    }

    void memset(int val)
    {
        MY_CUDA_CHECK_THROW(cudaMemset(d_ptr, val, sizeInBytes()));
    }
    void memsetAsync(int val)
    {
        MY_CUDA_CHECK_THROW(cudaMemsetAsync(d_ptr, val, sizeInBytes()));
    }

    T* data() { return d_ptr; }
    size_t size() { return n_elements; }
    size_t sizeInBytes() { return n_elements * sizeof(T); }


    bool readFile(std::vector<T> &vec, std::filesystem::path data_dir, const char filename[])
    {
        std::filesystem::path input_file = data_dir / std::filesystem::path(filename);
        std::ifstream stream(input_file.c_str(), std::ios::in | std::ios::binary);

        if (!stream)
            return false;

        stream.seekg(0, std::ios::end);
        size_t filesize = stream.tellg();
        stream.seekg(0, std::ios::beg);

        vec.resize(filesize / sizeof(T));
        stream.read((char *)vec.data(), filesize);

        return true;
    }

    bool readFileAndUpload(std::vector<T> &vec, std::filesystem::path data_dir, const char filename[])
    {
        bool exists = readFile(vec, data_dir, filename);
        if (exists)
            upload(vec);

        return exists;
    }

    bool readFileAndUpload(std::filesystem::path data_dir, const char filename[])
    {
        std::vector<T> vec;
        return readFileAndUpload(vec, data_dir, filename);
    }

    void write_to_file(std::filesystem::path output_dir, const char filename[])
    {
        std::filesystem::path output_file = output_dir / std::filesystem::path(filename);
        std::ofstream stream(output_file.c_str(), std::ios::out | std::ios::binary);

        std::vector<T> h_vec;
        download(h_vec);

        std::cout << filename << " : " << h_vec.size() << std::endl;
        stream.write((char *)h_vec.data(), h_vec.size() * sizeof(T));
    }

    void writeToBmpFile(std::function<unsigned char (T)> converter, std::filesystem::path output_dir, const char filename[], int w, int h)
    {
        std::filesystem::path output_file = output_dir / std::filesystem::path(filename);

        std::unique_ptr<T[]> image {std::make_unique<T[]>(w * h)};
        MY_CUDA_CHECK_THROW(cudaMemcpy(image.get(), this->d_ptr, w * h * sizeof(T), cudaMemcpyDeviceToHost));

        writeImageToBmpFile(image, output_file.c_str(), w, h, converter, true, true);
    }

private:
    T* d_ptr = nullptr;
    size_t n_elements = 0;
};

struct RenderBuffer
{
    RenderBuffer(int2 resolution) { _resolution = resolution; }

    virtual void resize(int2 resolution) = 0;
    virtual cudaSurfaceObject_t surface() = 0;
    
    int2 resolution() { return _resolution; }

protected:
    int2 _resolution{0, 0};
};