
#pragma once

#include "buffer.h"
#include "cuda_readwrite_array.h"

#include <string>
#include <filesystem>
#include <vector>

template <typename T>
struct DebugBuffer : public RenderBuffer
{
    DebugBuffer(int2 resolution);

    void resize(int2 resolution) override;

    void readFromFile(std::filesystem::path filename);
    void writeToFile(std::filesystem::path filename, std::function<unsigned char(T)> converter);

    void set(T val);

    cudaSurfaceObject_t surface() override { return _img_buffer.surface(); }
    cudaTextureObject_t texture() { return _img_buffer.texturePt(); }

private:
    ReadWriteCudaArray2D<T, false> _img_buffer;
};
