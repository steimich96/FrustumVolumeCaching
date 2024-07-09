

#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

#define STRINGIFY(x) #x
#define STR(x) STRINGIFY(x)
#define FILE_LINE __FILE__ ":" STR(__LINE__)

// CUDA_CHECK_THROW Already defined by TCNN
#define MY_CUDA_CHECK_THROW(x)                                                                                                        \
    do                                                                                                                             \
    {                                                                                                                              \
        cudaError_t result = x;                                                                                                    \
        if (result != cudaSuccess)                                                                                                 \
            throw std::runtime_error(std::string("CUDA | " FILE_LINE " " #x " failed with error: ") + cudaGetErrorString(result)); \
    } while (0)


#define CUDA_SYNC_CHECK_THROW()                                                                                                         \
    do                                                                                                                                  \
    {                                                                                                                                   \
        cudaStreamSynchronize(cudaStreamDefault);                                                                                                        \
        cudaError_t result = cudaGetLastError();                                                                                        \
        if (result != cudaSuccess)                                                                                                      \
            throw std::runtime_error(std::string("CUDA | " FILE_LINE " Synchronize failed with error: ") + cudaGetErrorString(result)); \
    } while (0)

#define CUDA_SYNC_CHECK_THROW_ASYNC(s)                                                                                                         \
    do                                                                                                                                  \
    {                                                                                                                                   \
        cudaStreamSynchronize(s);                                                                                                        \
        cudaError_t result = cudaGetLastError();                                                                                        \
        if (result != cudaSuccess)                                                                                                      \
            throw std::runtime_error(std::string("CUDA | " FILE_LINE " Synchronize failed with error: ") + cudaGetErrorString(result)); \
    } while (0)

inline void cudaPrintMemInfo()
{
    size_t free_byte, total_byte;
    cudaMemGetInfo(&free_byte, &total_byte);

    double free_db = (double)free_byte;
    double total_db = (double)total_byte;
    double used_db = total_db - free_db;

    std::cout << "GPU memory usage: used = " << used_db / 1024.0 / 1024.0 << " MB"
                              << ", free = " << free_db / 1024.0 / 1024.0 << " MB"
                              << ", total = " << total_db / 1024.0 / 1024.0 << " MB" << std::endl;
}

struct SimpleCudaTimer
{

    SimpleCudaTimer()
    {
        cudaEventCreate(&_start);
        cudaEventCreate(&_stop);
    }

    ~SimpleCudaTimer()
    {
        cudaEventDestroy(_start);
        cudaEventDestroy(_stop);
    }

    void start()
    {
        cudaEventRecord(_start, cudaStreamDefault);
    }

    void stop()
    {
        cudaEventRecord(_stop, cudaStreamDefault);
    }

    float elapsed()
    {
        float elapsed;
        cudaEventSynchronize(_stop);
        cudaEventElapsedTime(&elapsed, _start, _stop);
        return elapsed;
    }

    float stopElapsed()
    {
        stop();
        return elapsed();
    }

private:
    cudaEvent_t _start;
    cudaEvent_t _stop;
};