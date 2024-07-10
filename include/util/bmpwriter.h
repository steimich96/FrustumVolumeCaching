/*
 * Copyright (C) 2024, Michael Steiner, Graz University of Technology.
 * This code is licensed under the MIT license.
 */

#pragma once

#include "bmp.h"

#include <functional>
#include <memory>
#include <string>
#include <cmath>

template <typename T>
std::function<unsigned char(T)> createDirectConverter()
{
    return [](T x) -> unsigned char
    { return x; };
}

template <typename T>
std::function<unsigned char(T)> createMinMaxConverter(T min_val, T max_val)
{
    return [min_val, max_val](T x) -> unsigned char
    { return (unsigned char)((float)(min(max(x, min_val), max_val) - min_val) / (max_val - min_val) * 255.0f); };
}

template <typename T>
std::function<unsigned char(T)> createThresholdMaskConverter(T threshold)
{
    return [t = threshold](T x) -> unsigned char
    { return (unsigned char)(x > t) * 255; };
}

template <typename T>
std::function<unsigned char(T)> createNotZeroConverter()
{
    return [](T x) -> unsigned char
    { return (unsigned char)(x != 0) * 255; };
}

template <typename T>
std::function<unsigned char(T)> createIsZeroConverter()
{
    return [](T x) -> unsigned char
    { return (unsigned char)(x == 0) * 255; };
}

template <typename T>
void writeImageToBmpFile(std::unique_ptr<T[]> &image, const std::string file_name,
                         int width, int height, std::function<unsigned char(T)> converter,
                         bool single_channel = false, bool flip_image_horizontal = false)
{
    int w = width;
    int h = height;

    std::unique_ptr<unsigned char[]> out_image{std::make_unique<unsigned char[]>(width * height * 3)};

    // writing pixel data
    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            int in_i = flip_image_horizontal ? ((h - y - 1) * w + x) : (y * w + x);
            int out_i = y * w + x;

            if (single_channel)
            {
                unsigned char val = converter(image[in_i]);
                out_image[out_i * 3 + 0] = val;
                out_image[out_i * 3 + 1] = val;
                out_image[out_i * 3 + 2] = val;
            }
            else
            {
                out_image[out_i * 3 + 0] = converter(image[in_i * 4 + 2]);
                out_image[out_i * 3 + 1] = converter(image[in_i * 4 + 1]);
                out_image[out_i * 3 + 2] = converter(image[in_i * 4 + 0]);
            }
        }
    }

    generateBitmapImage(out_image.get(), height, width, file_name.c_str());
}