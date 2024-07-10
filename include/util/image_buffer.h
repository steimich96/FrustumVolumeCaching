/*
 * Copyright (C) 2024, Michael Steiner, Graz University of Technology.
 * This code is licensed under the MIT license.
 */

#pragma once

#include "buffer.h"
#include "cuda_readwrite_array.h"

#include <string>
#include <filesystem>
#include <vector>

struct ImageBuffer : public RenderBuffer
{
    ImageBuffer(int2 resolution);

    void resize(int2 resolution) override;

    float computePSNR(ImageBuffer& ref_image);
    float computePSNR(ImageBuffer& ref_image, bool write_diff_image, std::filesystem::path diff_img_filename);

    float computeWeightedPSNR(ImageBuffer& ref_image, ImageBuffer& weight_image);
    float computeWeightedPSNR(ImageBuffer& ref_image, ImageBuffer& weight_image, bool write_diff_image, std::filesystem::path diff_img_filename);

    void readFromFile(std::filesystem::path filename);
    void writeToFile(std::filesystem::path filename);

    cudaSurfaceObject_t surface() override { return _img_buffer.surface(); }
    cudaTextureObject_t texture() { return _img_buffer.texturePt(); }

private:
    void writeDiffImgToFile(std::filesystem::path diff_img_filename);

    ReadWriteCudaArray2D<uchar4, false>_img_buffer;

    CudaBuffer<float> _mse_buffer;
    CudaBuffer<float> _weight_buffer;
    CudaBuffer<unsigned char> _error_char_buffer;
    CudaBuffer<uchar4> _error_char4_buffer;
};