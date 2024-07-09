#include "interop_buffer.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <stb_image/stb_image.h>
#include <stb_image/stb_image_write.h>

InteropBuffer::InteropBuffer(int2 resolution) : RenderBuffer(resolution)
{
  frame_buffer[0] = 0;
  frame_buffer[1] = 0;
  render_buffer[0] = 0;
  render_buffer[1] = 0;

  d_frame_buffer_resource[0] = nullptr;
  d_frame_buffer_resource[1] = nullptr;
  d_frame_buffer_array[0] = nullptr;
  d_frame_buffer_array[1] = nullptr;
  
  d_frame_buffer_surface[0] = 0;
  d_frame_buffer_surface[1] = 0;

  init(resolution);
}

InteropBuffer::~InteropBuffer()
{

}

void InteropBuffer::destroy()
{
  for (int i = 0; i < 2; i++) 
  {
    if (d_frame_buffer_resource[i] != nullptr)
      MY_CUDA_CHECK_THROW(cudaGraphicsUnregisterResource(d_frame_buffer_resource[i]));

    MY_CUDA_CHECK_THROW(cudaDestroySurfaceObject(d_frame_buffer_surface[i]));
    //MY_CUDA_CHECK_THROW(cudaFreeArray(d_frame_buffer_array[i]));
  }

  if(render_buffer[0] != 0)
    glDeleteRenderbuffers(2, render_buffer);
  if (frame_buffer[0] != 0)
    glDeleteFramebuffers(2, frame_buffer);
}

bool InteropBuffer::init(int2 resolution)
{
  idx = 0;
  
  glCreateRenderbuffers(2, &(render_buffer[0]));
  glCreateFramebuffers(2, &(frame_buffer[0]));

  glNamedFramebufferRenderbuffer(frame_buffer[0], GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER,  render_buffer[0]);
  glNamedFramebufferRenderbuffer(frame_buffer[1], GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER,  render_buffer[1]);
  this->resize(resolution);

  d_current_buffer = d_frame_buffer_array[idx];
  d_current_buffer_surface = d_frame_buffer_surface[idx];

  return true;
}

void InteropBuffer::resize(int2 resolution)
{
  _resolution = resolution;

  if (!inference_mappings.empty())
    inference_mappings.clear();

  for (int i = 0; i < 2; i++) 
  {
    if (d_frame_buffer_resource[i] != NULL)
      MY_CUDA_CHECK_THROW(cudaGraphicsUnregisterResource(d_frame_buffer_resource[i]));

    glNamedRenderbufferStorage(render_buffer[i], GL_RGBA8, resolution.x, resolution.y);
    
    MY_CUDA_CHECK_THROW(cudaGraphicsGLRegisterImage(&(d_frame_buffer_resource[i]), render_buffer[i], GL_RENDERBUFFER,
      cudaGraphicsRegisterFlagsSurfaceLoadStore |
      cudaGraphicsRegisterFlagsWriteDiscard));
  }

  cudaGraphicsMapResources(2, &(d_frame_buffer_resource[0]), 0);
  for (int index = 0; index < 2; index++)
  {
    MY_CUDA_CHECK_THROW(cudaGraphicsSubResourceGetMappedArray(&(d_frame_buffer_array[index]), d_frame_buffer_resource[index], 0, 0));
    std::vector<void*> map;
    map.push_back((void*)&(d_frame_buffer_array[index]));
    inference_mappings.push_back(map);

    if (d_frame_buffer_surface[index])
      cudaDestroySurfaceObject(d_frame_buffer_surface[index]);

    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = d_frame_buffer_array[index];
    MY_CUDA_CHECK_THROW(cudaCreateSurfaceObject(&d_frame_buffer_surface[index], &resDesc));
  }
  MY_CUDA_CHECK_THROW(cudaGraphicsUnmapResources(2, &d_frame_buffer_resource[0], 0));
  swap();
  swap();
}

void InteropBuffer::blit(unsigned int target_width, unsigned int target_height)
{
  glBlitNamedFramebuffer(frame_buffer[idx], 0, 0, 0, _resolution.x, _resolution.y, 0, target_height, target_width, 0, GL_COLOR_BUFFER_BIT, target_width > _resolution.x ? GL_LINEAR : GL_NEAREST);
}

void InteropBuffer::clear()
{
  GLfloat clear_color[] = { 1.0f, 0.0f, 1.0f, 1.0f };
  glClearNamedFramebufferfv(frame_buffer[idx], GL_COLOR, 0, clear_color);
}

unsigned int InteropBuffer::swap()
{
  idx = (idx + 1) % 2;
  
  d_current_buffer = d_frame_buffer_array[idx];
  d_current_buffer_surface = d_frame_buffer_surface[idx];

  return idx;
}

void InteropBuffer::map()
{
  MY_CUDA_CHECK_THROW(cudaGraphicsMapResources(1, &d_frame_buffer_resource[idx]));
}

void InteropBuffer::unmap()
{
  MY_CUDA_CHECK_THROW(cudaGraphicsUnmapResources(1, &d_frame_buffer_resource[idx]));
}

void InteropBuffer::writeToFile(fs::path filename)
{
    int w = _resolution.x;
    int h = _resolution.y;

    std::vector<uchar4> image(w * h);
    MY_CUDA_CHECK_THROW(cudaMemcpy2DFromArray(image.data(), w * sizeof(uchar4), d_current_buffer, 0, 0, w * sizeof(uchar4), h, cudaMemcpyDeviceToHost));

    stbi_write_png(filename.string().c_str(), w, h, 4, image.data(), w * 4);
}