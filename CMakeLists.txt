
cmake_minimum_required(VERSION 3.21)
project(caching-nerf-project LANGUAGES C CXX CUDA)

option(RTX_ENABLED "Enables RTX kernels; Requires Optix installation" OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G")
elseif(UNIX)
    set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-Xcompiler -D_FORCE_INLINES -DVERBOSE --expt-extended-lambda -use_fast_math --expt-relaxed-constexpr -keep --ptxas-options=-v -lineinfo")
endif()

set(DEPENDENCIES_DIR "${CMAKE_SOURCE_DIR}/dependencies")
list(APPEND DEPENDENCIES_INCLUDE_DIRS "${DEPENDENCIES_DIR}/json")
list(APPEND DEPENDENCIES_INCLUDE_DIRS "${DEPENDENCIES_DIR}/stb_image")
list(APPEND DEPENDENCIES_INCLUDE_DIRS "${DEPENDENCIES_DIR}/argparse")
list(APPEND DEPENDENCIES_INCLUDE_DIRS "${DEPENDENCIES_DIR}/tiny_cuda_nn/include")

add_subdirectory("${DEPENDENCIES_DIR}/zstr")

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

add_subdirectory(dependencies/glfw)
add_subdirectory(dependencies/glad)
add_subdirectory(dependencies/glm)

set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/imgui)
add_library(imgui STATIC)
target_sources( imgui
                PRIVATE
                    ${IMGUI_DIR}/imgui_demo.cpp
                    ${IMGUI_DIR}/imgui_draw.cpp
                    ${IMGUI_DIR}/imgui_tables.cpp
                    ${IMGUI_DIR}/imgui_widgets.cpp
                    ${IMGUI_DIR}/imgui.cpp

                PRIVATE
                    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
                    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
                )

target_include_directories( imgui
                            PUBLIC ${IMGUI_DIR}
                            PUBLIC ${IMGUI_DIR}/backends
                            PUBLIC ${SDL3_DIR}/include
                            )

add_library(imgui_file_dialog STATIC)
target_sources(imgui_file_dialog PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/ImGuiFileDialog/ImGuiFileDialog.cpp)
target_include_directories(imgui_file_dialog PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/ImGuiFileDialog)
target_link_libraries(imgui_file_dialog PRIVATE imgui)

set(TCNN_BUILD_BENCHMARK OFF)
set(TCNN_BUILD_EXAMPLES OFF)
set(TCNN_ALLOW_CUBLAS_CUSOLVER OFF)
add_subdirectory(dependencies/tiny-cuda-nn)

set(CMAKE_CUDA_ARCHITECTURES ${TCNN_CUDA_ARCHITECTURES})   

include_directories("include")

if (RTX_ENABLED)
    set_property(SOURCE ${init_ray_payloads_ptx} PROPERTY SKIP_AUTOMOC ON)
    set_property(SOURCE ${sample_segments_ptx} PROPERTY SKIP_AUTOMOC ON)

    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${owl_dir}/owl/cmake/")
    add_subdirectory(${PROJECT_SOURCE_DIR}/dependencies/owl EXCLUDE_FROM_ALL)

    embed_ptx(OUTPUT_TARGET init_ray_payloads_ptx
            PTX_LINK_LIBRARIES owl::owl tiny-cuda-nn
            SOURCES src/programs/init_ray_payloads.cu)
    embed_ptx(OUTPUT_TARGET sample_segments_ptx
            PTX_LINK_LIBRARIES owl::owl tiny-cuda-nn
            SOURCES src/programs/sample_segments.cu)
endif()

add_library(caching-nerf STATIC)
target_sources(caching-nerf PRIVATE
    src/loader.cu
    src/basic_renderer.cu
    src/cache.cu
    src/cache_renderer.cu
    src/cache_basic_renderer.cu
    src/cache_performance_renderer.cu
    src/mesher.cu
    src/my_fully_fused_mlp.cu
    src/performance_renderer.cu
    src/renderer.cu
    src/util/debug_buffer.cu
    src/util/image_buffer.cu)

set_target_properties(caching-nerf PROPERTIES CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})

target_compile_options(caching-nerf PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--default-stream per-thread>)

target_include_directories(caching-nerf PUBLIC ${DEPENDENCIES_INCLUDE_DIRS} "include" "dependencies/pcg32")
target_link_libraries(caching-nerf PUBLIC 
    tiny-cuda-nn 
    zstr::zstr)

if (RTX_ENABLED)
    add_definitions(-DRTX_ENABLED)
    target_link_libraries(caching-nerf PUBLIC 
        owl::owl 
        init_ray_payloads_ptx 
        sample_segments_ptx)
endif()

add_subdirectory(realtime_viewer)

add_executable(eval src/eval.cu)
set_target_properties(eval PROPERTIES CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
target_link_libraries(eval PUBLIC caching-nerf glm)