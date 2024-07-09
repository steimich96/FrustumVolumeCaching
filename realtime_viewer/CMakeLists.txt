add_executable(realtime-viewer src/realtime_viewer.cu)
set_target_properties(realtime-viewer PROPERTIES CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
target_sources(realtime-viewer PRIVATE
    src/window.cu
    src/application.cu
    src/trackball_camera.cu
    src/interop_buffer.cu
    src/camera_path.cu)

target_compile_options(realtime-viewer PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--default-stream per-thread>)
target_include_directories(realtime-viewer PRIVATE "include")
target_link_libraries(realtime-viewer PUBLIC caching-nerf glad glfw glm::glm imgui imgui_file_dialog tiny-cuda-nn)
