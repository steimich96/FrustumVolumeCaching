/*
 * Copyright (C) 2024, Michael Steiner, Graz University of Technology.
 * This code is licensed under the MIT license.
 */

#include "util/debug_buffer.h"

#include "util/cub_helper.h"
#include "util/cuda_helper.h"
#include "util/helper_math.h"

// #define STB_IMAGE_IMPLEMENTATION
// #define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image/stb_image.h>
#include <stb_image/stb_image_write.h>

#include <cstring>

namespace fs = std::filesystem;

template <typename T>
__global__ void set_debug_buffer_kernel(int2 resolution, cudaSurfaceObject_t surf, T val)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= resolution.x || y >= resolution.y)
        return;

    surf2Dwrite(val, surf, x * sizeof(T), y);
}

template <typename T>
DebugBuffer<T>::DebugBuffer(int2 resolution) : RenderBuffer(resolution)
{
    resize(resolution);
}

template <typename T>
void DebugBuffer<T>::resize(int2 resolution)
{
    int n_pixels = resolution.x * resolution.y;
    _img_buffer.resize(resolution);
}

template <typename T>
void DebugBuffer<T>::set(T val)
{    
    const int BLOCK_SIZE_2D = 16;
    dim3 block_size_2D(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
    dim3 image_grid_size_2D(toDim3(divRoundUp(_resolution, BLOCK_SIZE_2D)));
    set_debug_buffer_kernel<<<image_grid_size_2D, block_size_2D>>>(_resolution, _img_buffer.surface(), val);
}

template <typename T>
void DebugBuffer<T>::readFromFile(fs::path filename)
{
	int comp = 0;
    int2 image_resolution;
    T* h_img = (T*) stbi_load(filename.string().c_str(), &image_resolution.x, &image_resolution.y, &comp, 1);

    if (image_resolution.x != _resolution.x || image_resolution.y != _resolution.y)
        throw std::runtime_error("Image resolution has to be set in advance (currently)");

    _img_buffer.upload(h_img);
    free(h_img);
}

template <typename T>
void DebugBuffer<T>::writeToFile(fs::path filename, std::function<unsigned char(T)> converter)
{
    int w = _resolution.x;
    int h = _resolution.y;

    std::vector<T> image(w * h);
    _img_buffer.download(image.data());

    std::vector<unsigned char> out_image_char(w*h);
    std::transform(image.begin(), image.end(), out_image_char.begin(), converter);

    if (filename.extension() == ".png") {
        stbi_write_png(filename.string().c_str(), w, h, 1, out_image_char.data(), w * 1);
    } else if (filename.extension() == ".bmp") {
        stbi_write_bmp(filename.string().c_str(), w, h, 1, out_image_char.data());
    } else if (filename.extension() == ".jpg") {
        stbi_write_jpg(filename.string().c_str(), w, h, 1, out_image_char.data(), 100);
    } else {
        throw std::runtime_error("Image file extension not supported!");
    }
}

template struct DebugBuffer<float>;