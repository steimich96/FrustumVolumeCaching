
#include "util/image_buffer.h"

#include "util/bmpwriter.h"
#include "util/cub_helper.h"
#include "util/cuda_helper.h"
#include "util/helper_math.h"

// #define STB_IMAGE_IMPLEMENTATION
// #define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image/stb_image.h>
#include <stb_image/stb_image_write.h>

#include <cstring>

namespace fs = std::filesystem;


inline __device__ __host__ uint to_8bit(const float f)
{
    return min(255, max(0, int(f * 256.f)));
}
inline __device__ __host__ uchar4 color_to_uchar4(const float3 color)
{
    uchar4 data;
    data.x = to_8bit(color.x);
    data.y = to_8bit(color.y);
    data.z = to_8bit(color.z);
    data.w = 255U;
    return data;
}
inline __device__ __host__ uchar4 applyBackground(uchar4 original, float3 background_color)
{
    float3 original_color = make_float3(original.x, original.y, original.z) / 255.f;
    float alpha = ((float)original.w) / 255.f;

    return color_to_uchar4(original_color * alpha + (1.0f - alpha) * background_color);
}

__device__ __host__ inline float mse_to_psnr(float mse)
{
    return -10.0f * std::log(mse) / std::log(10.0f);
}

__device__ __host__ inline float compute_mse(uchar4 px_img1, uchar4 px_img2)
{
    float3 diff = (make_float3(px_img1.x, px_img1.y, px_img1.z) - make_float3(px_img2.x, px_img2.y, px_img2.z)) / 255.0f;
    return dot(diff, diff) / 3.0f;
}

__device__ __host__ inline float compute_error(uchar4 px_img1, uchar4 px_img2)
{
    float3 diff = (make_float3(px_img1.x, px_img1.y, px_img1.z) - make_float3(px_img2.x, px_img2.y, px_img2.z)) / 255.0f;
    return length(diff);
}

__device__ float3 colormapMagma(float x)
{
	float3 c0 = make_float3(-0.002136485053939582, -0.000749655052795221, -0.005386127855323933);
	float3 c1 = make_float3(0.2516605407371642, 0.6775232436837668, 2.494026599312351);
	float3 c2 = make_float3(8.353717279216625, -3.577719514958484, 0.3144679030132573);
	float3 c3 = make_float3(-27.66873308576866, 14.26473078096533, -13.64921318813922);
	float3 c4 = make_float3(52.17613981234068, -27.94360607168351, 12.94416944238394);
	float3 c5 = make_float3(-50.76852536473588, 29.04658282127291, 4.23415299384598);
	float3 c6 = make_float3(18.65570506591883, -11.48977351997711, -5.601961508734096);
	x = clamp(x, 0.f, 1.f);
	float3 res = (c0+x*(c1+x*(c2+x*(c3+x*(c4+x*(c5+c6*x))))));
	return clamp(res, 0.0f, 1.0f);
}

__global__ void computeTex2Tex_mse_kernel(int2 resolution, cudaTextureObject_t tex_img1, cudaTextureObject_t tex_img2, float* mse_buffer, unsigned char* error_char_buffer, uchar4* error_char4_buffer)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= resolution.x || y >= resolution.y)
        return;

    const int px_idx = y * resolution.x + x;

    uchar4 px_img1 = tex2D<uchar4>(tex_img1, x, y);
    uchar4 px_img2 = tex2D<uchar4>(tex_img2, x, y);

    mse_buffer[px_idx] = compute_mse(px_img1, px_img2);
    error_char_buffer[px_idx] = clamp(compute_error(px_img1, px_img2), 0.0f, 1.0f) * 255.f;

    float3 error_colored = colormapMagma(compute_error(px_img1, px_img2)) * 255.f;
    error_char4_buffer[px_idx] = make_uchar4(error_colored.x, error_colored.y, error_colored.z, 255U);
}
__global__ void computeTex2Tex_weighted_mse_kernel(int2 resolution, cudaTextureObject_t tex_img1, cudaTextureObject_t tex_img2, cudaTextureObject_t tex_weights,
                                                   float *mse_buffer, float *weight_buffer, unsigned char *error_char_buffer, uchar4* error_char4_buffer)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= resolution.x || y >= resolution.y)
        return;

    const int px_idx = y * resolution.x + x;

    uchar4 px_img1 = tex2D<uchar4>(tex_img1, x, y);
    uchar4 px_img2 = tex2D<uchar4>(tex_img2, x, y);
    float weight = tex2D<uchar4>(tex_weights, x, y).x / 255.f;

    mse_buffer[px_idx] = compute_mse(px_img1, px_img2) * weight;
    weight_buffer[px_idx] = weight;
    error_char_buffer[px_idx] = clamp(compute_error(px_img1, px_img2), 0.0f, 1.0f) * 255.f * weight;

    float3 error_colored = colormapMagma(compute_error(px_img1, px_img2)) * 255.f * weight;
    error_char4_buffer[px_idx] = make_uchar4(error_colored.x, error_colored.y, error_colored.z, 255U);
}

__global__ void applyBackground_kernel(int2 resolution, cudaSurfaceObject_t surf, float3 background_color)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= resolution.x || y >= resolution.y)
        return;

    uchar4 px = surf2Dread<uchar4>(surf, x * sizeof(uchar4), y);
    surf2Dwrite(applyBackground(px, background_color), surf, x * sizeof(uchar4), y);
}

void writeImageToFile(void* image, int w, int h, int channels, fs::path filename)
{
    if (filename.extension() == ".png") {
        stbi_write_png(filename.string().c_str(), w, h, channels, image, w * channels);
    } else if (filename.extension() == ".bmp") {
        stbi_write_bmp(filename.string().c_str(), w, h, channels, image);
    } else if (filename.extension() == ".jpg") {
        stbi_write_jpg(filename.string().c_str(), w, h, channels, image, 100);
    } else {
        throw std::runtime_error("Image file extension not supported!");
    }
}

ImageBuffer::ImageBuffer(int2 resolution) : RenderBuffer(resolution)
{
    resize(resolution);
}

void ImageBuffer::resize(int2 resolution)
{
    int n_pixels = resolution.x * resolution.y;
    _img_buffer.resize(resolution);

    _mse_buffer.resize(n_pixels);
    _weight_buffer.resize(n_pixels);
    _error_char_buffer.resize(n_pixels);
    _error_char4_buffer.resize(n_pixels);
}

void ImageBuffer::readFromFile(fs::path filename)
{
	int comp = 0;
    int2 image_resolution;
    uchar4* h_img = (uchar4*) stbi_load(filename.string().c_str(), &image_resolution.x, &image_resolution.y, &comp, 4);

    if (image_resolution.x != _resolution.x || image_resolution.y != _resolution.y)
        throw std::runtime_error("Image resolution has to be set in advance (currently)");

    _img_buffer.upload(h_img);
    free(h_img);

    const int BLOCK_SIZE_2D = 16;
    dim3 block_size_2D(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
    dim3 image_grid_size_2D(toDim3(divRoundUp(_resolution, BLOCK_SIZE_2D)));
    applyBackground_kernel<<<image_grid_size_2D, block_size_2D>>>(_resolution, _img_buffer.surface(), make_float3(1.0f));
    CUDA_SYNC_CHECK_THROW();
}

void ImageBuffer::writeToFile(fs::path filename)
{
    int w = _resolution.x;
    int h = _resolution.y;

    std::vector<uchar4> image(w * h);
    _img_buffer.download(image.data());

    writeImageToFile(image.data(), w, h, 4, filename);
}

float ImageBuffer::computePSNR(ImageBuffer& ref_image)
{
    if (ref_image.resolution().x != _resolution.x || ref_image.resolution().y != _resolution.y)
        throw std::runtime_error("PSNR computation only supported for same size images");

    int n_pixels = _resolution.x * _resolution.y;

    const int BLOCK_SIZE_2D = 16;
    dim3 block_size_2D(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
    dim3 image_grid_size_2D(toDim3(divRoundUp(_resolution, BLOCK_SIZE_2D)));
    computeTex2Tex_mse_kernel<<<image_grid_size_2D, block_size_2D>>>(_resolution, this->texture(), ref_image.texture(),
                                                                     _mse_buffer.data(), _error_char_buffer.data(), _error_char4_buffer.data());
    CUDA_SYNC_CHECK_THROW();

    float mse = cubGetDeviceSum<float, float>(_mse_buffer.data(), n_pixels) / n_pixels;
    return mse_to_psnr(mse);
}

float ImageBuffer::computeWeightedPSNR(ImageBuffer& ref_image, ImageBuffer& weight_image)
{
    if (ref_image.resolution().x != _resolution.x || ref_image.resolution().y != _resolution.y)
        throw std::runtime_error("PSNR computation only supported for same size images");

    int n_pixels = _resolution.x * _resolution.y;

    const int BLOCK_SIZE_2D = 16;
    dim3 block_size_2D(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
    dim3 image_grid_size_2D(toDim3(divRoundUp(_resolution, BLOCK_SIZE_2D)));
    computeTex2Tex_weighted_mse_kernel<<<image_grid_size_2D, block_size_2D>>>(_resolution, this->texture(), ref_image.texture(), weight_image.texture(),
                                                                              _mse_buffer.data(), _weight_buffer.data(), _error_char_buffer.data(), _error_char4_buffer.data());
    CUDA_SYNC_CHECK_THROW();

    float weight_sum = cubGetDeviceSum<float, float>(_weight_buffer.data(), n_pixels);
    float weighted_mse = cubGetDeviceSum<float, float>(_mse_buffer.data(), n_pixels) / weight_sum;
    return mse_to_psnr(weighted_mse);
}

float ImageBuffer::computePSNR(ImageBuffer& ref_image, bool write_diff_image, fs::path diff_img_filename)
{
    float psnr = computePSNR(ref_image);
    if (write_diff_image) writeDiffImgToFile(diff_img_filename);
    return psnr;
}

float ImageBuffer::computeWeightedPSNR(ImageBuffer& ref_image, ImageBuffer& weight_image, bool write_diff_image, std::filesystem::path diff_img_filename)
{
    float psnr = computeWeightedPSNR(ref_image, weight_image);
    if (write_diff_image) writeDiffImgToFile(diff_img_filename);
    return psnr;
}

void ImageBuffer::writeDiffImgToFile(fs::path filename)
{
    std::vector<uchar4> h_error_buffer;
    _error_char4_buffer.download(h_error_buffer);

    writeImageToFile((unsigned char*) h_error_buffer.data(), _resolution.x, _resolution.y, 4, filename);
}