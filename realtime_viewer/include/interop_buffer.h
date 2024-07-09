#pragma once

#include <util/buffer.h>

#include <glad/glad.h>
#include <cuda_gl_interop.h>

#include <filesystem>
namespace fs = std::filesystem;


class InteropBuffer : public RenderBuffer
{
private:
  GLuint frame_buffer[2];
  GLuint render_buffer[2];

  cudaGraphicsResource_t d_frame_buffer_resource[2];
  cudaArray_t d_frame_buffer_array[2];
  cudaSurfaceObject_t d_frame_buffer_surface[2];

  cudaArray_t d_current_buffer = nullptr;
  cudaSurfaceObject_t d_current_buffer_surface = 0;

  unsigned int idx;

  std::vector <std::vector<void*>> inference_mappings;

  bool init(int2 resolution);

public:
  InteropBuffer(int2 resolution);
  ~InteropBuffer();
  void destroy();

  cudaSurfaceObject_t surface() override { return d_current_buffer_surface; }

  void resize(int2 resolution) override;
  unsigned int swap();
  void blit(unsigned int target_width, unsigned int target_height);
  void clear();
  void map();
  void unmap();
  void writeToFile(fs::path filename);
};