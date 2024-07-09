
// Code partially adapted from https://github.com/owl-project/owlDVR_SpaceSkip/blob/master/SpaceSkipper.cu

#include "mesher.h"

#include "raymarch_common.h"
#include "renderer_common.h"
#include "util/cub_helper.h"
#include "util/cuda_helper.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

using namespace cooperative_groups;
namespace cg = cooperative_groups;

#include <fstream>
#include <sstream>
namespace fs = std::filesystem;

__global__ void initVertices(const SceneInfo scene_info, const int vertice_per_lvl, float3* vertices, const int* vertex_is_active, const int* vertex_idx_buffer)
{
    uint3 vertex_idx_3D = threadIdx + blockIdx * make_uint3(blockDim.x, blockDim.y, blockDim.z);

    if (!gridContains3D(vertex_idx_3D, scene_info.grid_resolution + 1))
        return;

    const float3 aabb_center = (scene_info.aabb_from + scene_info.aabb_to) * 0.5f;

    for (int lvl = 0; lvl < scene_info.grid_nlvl; lvl++)
    {
        const float lvl_scale = 1 << (scene_info.grid_nlvl - lvl - 1);

        const float3 volume_dims = (scene_info.aabb_to - scene_info.aabb_from) / lvl_scale;
        const float3 vertex_pos = aabb_center - volume_dims * 0.5f + make_float3(vertex_idx_3D) / (float) scene_info.grid_resolution * volume_dims;
        const int vertex_idx_1D = to1DMulti(make_int3(vertex_idx_3D), scene_info.grid_resolution + 1, lvl, vertice_per_lvl);

        int vertex_idx = vertex_is_active != nullptr ? vertex_idx_buffer[vertex_idx_1D] : vertex_idx_1D;

        if (vertex_is_active == nullptr || vertex_is_active[vertex_idx_1D])
            vertices[vertex_idx] = vertex_pos;
    }
}

template<int DIM> inline __device__ int3 prev_voxel(const int3 idx);
template<int DIM> inline __device__ int3 vertexIdx_du();
template<int DIM> inline __device__ int3 vertexIdx_dv();

template<> inline __device__ int3 prev_voxel<0>(const int3 idx) { return idx - make_int3(1, 0, 0); }
template<> inline __device__ int3 prev_voxel<1>(const int3 idx) { return idx - make_int3(0, 1, 0); }
template<> inline __device__ int3 prev_voxel<2>(const int3 idx) { return idx - make_int3(0, 0, 1); }

template<> inline __device__ int3 vertexIdx_du<0>() { return { 0, 1, 0 }; }
template<> inline __device__ int3 vertexIdx_dv<0>() { return { 0, 0, 1 }; }

template<> inline __device__ int3 vertexIdx_du<1>() { return { 0, 0, 1 }; }
template<> inline __device__ int3 vertexIdx_dv<1>() { return { 1, 0, 0 }; }

template<> inline __device__ int3 vertexIdx_du<2>() { return { 1, 0, 0 }; }
template<> inline __device__ int3 vertexIdx_dv<2>() { return { 0, 1, 0 }; }

template<int WIDTH> inline __device__ uint3 to_3D_with_mortonXY(const uint3 idx)
{
    uint32_t temp = idx.y * WIDTH + idx.x;
    return make_uint3(morton2D_invert(temp >> 0), morton2D_invert(temp >> 1), idx.z);
}

// Move the DIM-coordinate back to the z-coordinate
template<int DIM> inline __host__ __device__ uint3 DIM_to_z(const uint3 idx);
template<> inline __host__ __device__ uint3 DIM_to_z<0>(const uint3 idx) { return make_uint3(idx.z, idx.y, idx.x); }
template<> inline __host__ __device__ uint3 DIM_to_z<1>(const uint3 idx) { return make_uint3(idx.x, idx.z, idx.y); }
template<> inline __host__ __device__ uint3 DIM_to_z<2>(const uint3 idx) { return make_uint3(idx.y, idx.x, idx.z); }

template<int DIM> inline __host__ __device__ int3 DIM_to_z(const int3 idx) { return make_int3(DIM_to_z<DIM>(make_uint3(idx))); }
template<int DIM> inline __host__ __device__ dim3 DIM_to_z(const dim3 idx) { return toDim3(DIM_to_z<DIM>(make_uint3(idx.x, idx.y, idx.z))); }


template <typename T> 
inline __device__ bool is_active_multi(int3 idx, const int3 grid_dims, int lvl, const int max_lvl, const int vals_per_lvl, const T *is_active_grid)
{
    if (!gridContains3D(idx, grid_dims))
    {
        if (lvl >= (max_lvl - 1))
            return false;

        lvl += 1;
        idx = (idx + grid_dims / 2) / 2;
    }

    return grid_occupied_at<DEFAULT_GRID_RESOLUTION, GRID_IS_MORTON>(idx, lvl, is_active_grid);
}

template <int DIM, int BLOCK_SIZE, typename T>
__global__ void createFacesAlongDimMulti(const int3 grid_dims,
                                         const int lvl,
                                         const int max_lvl,
                                         const int vals_per_lvl,
                                         const int vertices_per_lvl,

                                         bool first_pass,
                                         const int *vertices_idcs,
                                         int3 *out_indices,
                                         int *out_active_vertices,

                                         int *d_triangle_counter,
                                         const T *is_active_grid)
{
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<2> tile2 = cg::tiled_partition<2>(block);
    cg::thread_block_tile<4> tile4 = cg::tiled_partition<4>(block);
    cg::thread_block_tile<8> tile8 = cg::tiled_partition<8>(block);
    cg::thread_block_tile<16> tile16 = cg::tiled_partition<16>(block);

    // X & Y coordinates should be in morton order (z-order) and DIM should be the last coordinate (z)
    const uint3 morton_threadIdx = to_3D_with_mortonXY<BLOCK_SIZE>(threadIdx);
    const uint3 vertex_idx_3D = DIM_to_z<DIM>(morton_threadIdx + blockIdx * make_uint3(blockDim.x, blockDim.y, blockDim.z));

    if (!gridContains3D(vertex_idx_3D, grid_dims + make_int3(DIM == 0, DIM == 1, DIM == 2)))
        return;

    const int3 idx1 = make_int3(vertex_idx_3D); // CURRENT index
    const int3 idx0 = prev_voxel<DIM>(idx1);    // index of PREVIOUS entry in given DIM

    if (lvl > 0 && (gridContains3D(idx1 - grid_dims / 4, grid_dims / 2) || gridContains3D(idx0 - grid_dims / 4, grid_dims / 2))) // ignore region of lower-level (higher resolution)
        return;

    const bool active0 = is_active_multi(idx0, grid_dims, lvl, max_lvl, vals_per_lvl, is_active_grid);
    const bool active1 = is_active_multi(idx1, grid_dims, lvl, max_lvl, vals_per_lvl, is_active_grid);

    if (active0 == active1) // NO BOUNDARY
        return;

    const int3 du = vertexIdx_du<DIM>();
    const int3 dv = vertexIdx_dv<DIM>();

    const int face_0_to_1 = active0 && !active1;
    const int face_1_to_0 = !face_0_to_1;

    const int tile2_all_aggree =  tile2.all(face_0_to_1) || tile2.all(face_1_to_0);
    const int tile4_all_aggree =  tile4.all(face_0_to_1) || tile4.all(face_1_to_0);
    const int tile8_all_aggree =  tile8.all(face_0_to_1) || tile8.all(face_1_to_0);
    const int tile16_all_aggree = tile16.all(face_0_to_1) || tile16.all(face_1_to_0);

    bool writing = true;
    int tu = 1;
    int tv = 1;
    if (tile16_all_aggree)
    {
        writing = tile16.thread_rank() == 0;
        tv = cg::reduce(tile16, morton_threadIdx.x, cg::greater<uint>()) - cg::reduce(tile16, morton_threadIdx.x, cg::less<uint>()) + 1;
        tu = cg::reduce(tile16, morton_threadIdx.y, cg::greater<uint>()) - cg::reduce(tile16, morton_threadIdx.y, cg::less<uint>()) + 1;
    }
    else if (tile8_all_aggree)
    {
        writing = tile8.thread_rank() == 0;
        tv = cg::reduce(tile8, morton_threadIdx.x, cg::greater<uint>()) - cg::reduce(tile8, morton_threadIdx.x, cg::less<uint>()) + 1;
        tu = cg::reduce(tile8, morton_threadIdx.y, cg::greater<uint>()) - cg::reduce(tile8, morton_threadIdx.y, cg::less<uint>()) + 1;
    }
    else if (tile4_all_aggree)
    {
        writing = tile4.thread_rank() == 0;
        tv = cg::reduce(tile4, morton_threadIdx.x, cg::greater<uint>()) - cg::reduce(tile4, morton_threadIdx.x, cg::less<uint>()) + 1;
        tu = cg::reduce(tile4, morton_threadIdx.y, cg::greater<uint>()) - cg::reduce(tile4, morton_threadIdx.y, cg::less<uint>()) + 1;
    }
    else if (tile2_all_aggree)
    {
        writing = tile2.thread_rank() == 0;
        tv = cg::reduce(tile2, morton_threadIdx.x, cg::greater<uint>()) - cg::reduce(tile2, morton_threadIdx.x, cg::less<uint>()) + 1;
        tu = cg::reduce(tile2, morton_threadIdx.y, cg::greater<uint>()) - cg::reduce(tile2, morton_threadIdx.y, cg::less<uint>()) + 1;
    }
    
    if (!writing)
        return;

    const int vtx00 = to1DMulti(idx1, grid_dims + 1, lvl, vertices_per_lvl);
    const int vtx01 = to1DMulti(idx1 + tu * du, grid_dims + 1, lvl, vertices_per_lvl);
    const int vtx10 = to1DMulti(idx1 + tv * dv, grid_dims + 1, lvl, vertices_per_lvl);
    const int vtx11 = to1DMulti(idx1 + tu * du + tv * dv, grid_dims + 1, lvl, vertices_per_lvl);

    int triangle_idx = atomicAdd(d_triangle_counter, 2);

    if (first_pass)
    {
        out_active_vertices[vtx00] = 1;
        out_active_vertices[vtx01] = 1;
        out_active_vertices[vtx10] = 1;
        out_active_vertices[vtx11] = 1;
    }
    else
    {
        const int vtx00_idx = vertices_idcs != nullptr ? vertices_idcs[vtx00] : vtx00;
        const int vtx01_idx = vertices_idcs != nullptr ? vertices_idcs[vtx01] : vtx01;
        const int vtx10_idx = vertices_idcs != nullptr ? vertices_idcs[vtx10] : vtx10;
        const int vtx11_idx = vertices_idcs != nullptr ? vertices_idcs[vtx11] : vtx11;

        if (face_0_to_1)
        {
            // active boundary, facing from idx0 to idx1
            out_indices[triangle_idx + 0] = {vtx00_idx, vtx01_idx, vtx11_idx};
            out_indices[triangle_idx + 1] = {vtx00_idx, vtx11_idx, vtx10_idx};
        }
        else
        {
            // active boundary, facing from idx1 to idx0
            out_indices[triangle_idx + 0] = {vtx01_idx, vtx00_idx, vtx11_idx};
            out_indices[triangle_idx + 1] = {vtx11_idx, vtx00_idx, vtx10_idx};
        }
    }
}

void createMeshFromGrid_firstPass(SceneInfo scene_info, CudaBuffer<uint8_t>& grid, CudaBuffer<int>& vertex_is_active_buffer, int& n_triangles, int& n_vertices)
{
    const int vals_per_lvl = pow(scene_info.grid_resolution, 3);
    const int vertices_per_level = pow(scene_info.grid_resolution + 1, 3);

    vertex_is_active_buffer.resize(vertices_per_level * scene_info.grid_nlvl);
    vertex_is_active_buffer.memset(0);

    
    constexpr unsigned int BLOCK_SIZE = 4;
    const unsigned int grid_size = divRoundUp(scene_info.grid_resolution + 1, BLOCK_SIZE);
    int3 grid_dims = make_int3(scene_info.grid_resolution);

    CudaBuffer<int> triangle_counter(1, 0);

    for (int lvl = 0; lvl < scene_info.grid_nlvl; lvl++)
    {
        createFacesAlongDimMulti<0, BLOCK_SIZE><<<toDim3(grid_size), toDim3(BLOCK_SIZE)>>>(grid_dims, lvl, scene_info.grid_nlvl, vals_per_lvl, vertices_per_level, true, nullptr, nullptr, vertex_is_active_buffer.data(), triangle_counter.data(), grid.data());
        createFacesAlongDimMulti<1, BLOCK_SIZE><<<toDim3(grid_size), toDim3(BLOCK_SIZE)>>>(grid_dims, lvl, scene_info.grid_nlvl, vals_per_lvl, vertices_per_level, true, nullptr, nullptr, vertex_is_active_buffer.data(), triangle_counter.data(), grid.data());
        createFacesAlongDimMulti<2, BLOCK_SIZE><<<toDim3(grid_size), toDim3(BLOCK_SIZE)>>>(grid_dims, lvl, scene_info.grid_nlvl, vals_per_lvl, vertices_per_level, true, nullptr, nullptr, vertex_is_active_buffer.data(), triangle_counter.data(), grid.data());
    }
    CUDA_SYNC_CHECK_THROW();

    n_triangles = triangle_counter.downloadFirst();
    n_vertices = cubGetDeviceSum<int, int>(vertex_is_active_buffer.data(), vertex_is_active_buffer.size());
}

void createMeshFromGrid_secondPass(SceneInfo scene_info, CudaBuffer<uint8_t>& grid, CudaBuffer<int>& vertex_is_active_buffer, float3* d_vertices, int3* d_indices)
{
    const int vals_per_lvl = pow(scene_info.grid_resolution, 3);
    const int vertices_per_level = pow(scene_info.grid_resolution + 1, 3);

    CudaBuffer<int> vertex_index_buffer(vertices_per_level * scene_info.grid_nlvl);
    cubDeviceExclusiveSum(vertex_is_active_buffer.data(), vertex_index_buffer.data(), vertex_index_buffer.size());


    constexpr unsigned int BLOCK_SIZE = 4;
    const unsigned int grid_size = divRoundUp(scene_info.grid_resolution + 1, BLOCK_SIZE);
    initVertices<<<toDim3(grid_size), toDim3(BLOCK_SIZE)>>>(scene_info, vertices_per_level, d_vertices, vertex_is_active_buffer.data(), vertex_index_buffer.data());
    CUDA_SYNC_CHECK_THROW();

    int3 grid_dims = make_int3(scene_info.grid_resolution);
    CudaBuffer<int> triangle_counter(1, 0);

    for (int lvl = 0; lvl < scene_info.grid_nlvl; lvl++)
    {
        createFacesAlongDimMulti<0, BLOCK_SIZE><<<toDim3(grid_size), toDim3(BLOCK_SIZE)>>>(grid_dims, lvl, scene_info.grid_nlvl, vals_per_lvl, vertices_per_level, false, vertex_index_buffer.data(), d_indices, nullptr, triangle_counter.data(), grid.data());
        createFacesAlongDimMulti<1, BLOCK_SIZE><<<toDim3(grid_size), toDim3(BLOCK_SIZE)>>>(grid_dims, lvl, scene_info.grid_nlvl, vals_per_lvl, vertices_per_level, false, vertex_index_buffer.data(), d_indices, nullptr, triangle_counter.data(), grid.data());
        createFacesAlongDimMulti<2, BLOCK_SIZE><<<toDim3(grid_size), toDim3(BLOCK_SIZE)>>>(grid_dims, lvl, scene_info.grid_nlvl, vals_per_lvl, vertices_per_level, false, vertex_index_buffer.data(), d_indices, nullptr, triangle_counter.data(), grid.data());
    }
    CUDA_SYNC_CHECK_THROW();
}

int createMeshFromGridSinglePass(float3* d_vertices, int3* d_indices, SceneInfo scene_info, CudaBuffer<uint8_t>& grid)
{
    const int vals_per_lvl = pow(scene_info.grid_resolution, 3);
    const int vertices_per_level = pow(scene_info.grid_resolution + 1, 3);

    constexpr unsigned int BLOCK_SIZE = 4;
    const unsigned int grid_size = divRoundUp(scene_info.grid_resolution + 1, BLOCK_SIZE);
    initVertices<<<toDim3(grid_size), toDim3(BLOCK_SIZE)>>>(scene_info, vertices_per_level, d_vertices, nullptr, nullptr);
    CUDA_SYNC_CHECK_THROW();

    int3 grid_dims = make_int3(scene_info.grid_resolution);
    CudaBuffer<int> triangle_counter(1, 0);

    for (int lvl = 0; lvl < scene_info.grid_nlvl; lvl++)
    {
        createFacesAlongDimMulti<0, BLOCK_SIZE><<<toDim3(grid_size), toDim3(BLOCK_SIZE)>>>(grid_dims, lvl, scene_info.grid_nlvl, vals_per_lvl, vertices_per_level, false, nullptr, d_indices, nullptr, triangle_counter.data(), grid.data());
        createFacesAlongDimMulti<1, BLOCK_SIZE><<<toDim3(grid_size), toDim3(BLOCK_SIZE)>>>(grid_dims, lvl, scene_info.grid_nlvl, vals_per_lvl, vertices_per_level, false, nullptr, d_indices, nullptr, triangle_counter.data(), grid.data());
        createFacesAlongDimMulti<2, BLOCK_SIZE><<<toDim3(grid_size), toDim3(BLOCK_SIZE)>>>(grid_dims, lvl, scene_info.grid_nlvl, vals_per_lvl, vertices_per_level, false, nullptr, d_indices, nullptr, triangle_counter.data(), grid.data());
    }
    CUDA_SYNC_CHECK_THROW();

    return triangle_counter.downloadFirst();
}

int createMeshFromGridCompact(TriangleMesh& mesh, SceneInfo scene_info, CudaBuffer<uint8_t>& grid)
{
    const int vertices_per_level = pow(scene_info.grid_resolution + 1, 3);

    CudaBuffer<int> vertex_is_active_buffer;
    int n_triangles, n_vertices;
    createMeshFromGrid_firstPass(scene_info, grid, vertex_is_active_buffer, n_triangles, n_vertices);

    mesh.vertices.resize(n_vertices);
    mesh.indices.resize(n_triangles);
    mesh.total_n_vertices = n_vertices;
    mesh.n_triangles = n_triangles;

    createMeshFromGrid_secondPass(scene_info, grid, vertex_is_active_buffer, mesh.vertices.data(), mesh.indices.data());
    return mesh.n_triangles;
}

void writeMeshToObj(std::vector<float3>& vertices, std::vector<int3> indices, int n_triangles, fs::path data_dir, const char filename[])
{
    fs::path output_file = data_dir / fs::path(filename);
    std::ofstream fstream(output_file.c_str(), std::ios::out);

    std::stringstream sstream;
    for (auto& v : vertices)
    {
        sstream << "v " << v.x << " " << v.y << " " << v.z << std::endl;
    }

    for (int i = 0; i < n_triangles; i++)
    {
        int3 index = indices[i];
        sstream << "f " << index.x + 1 << " " << index.y + 1 << " " << index.z + 1 << std::endl;
    }

    fstream << sstream.rdbuf();
}

void writeMeshToObj(TriangleMesh& mesh, fs::path data_dir, const char filename[])
{
    std::vector<float3> vertices;
    mesh.vertices.download(vertices);

    std::vector<int3> indices;
    mesh.indices.download(indices);

    writeMeshToObj(vertices, indices, mesh.n_triangles, data_dir, filename);
}

#ifdef RTX_ENABLED
int createMeshFromGridCompact(TriangleMeshOwl& mesh, SceneInfo scene_info, CudaBuffer<uint8_t>& grid, OWLContext context)
{
    const int vertices_per_level = pow(scene_info.grid_resolution + 1, 3);

    CudaBuffer<int> vertex_is_active_buffer;
    int n_triangles, n_vertices;
    createMeshFromGrid_firstPass(scene_info, grid, vertex_is_active_buffer, n_triangles, n_vertices);

    mesh.vertices = owlDeviceBufferCreate(context, OWL_FLOAT3, n_vertices, 0);
    mesh.indices = owlDeviceBufferCreate(context, OWL_INT3, n_triangles, 0);
    mesh.total_n_vertices = n_vertices;
    mesh.n_triangles = n_triangles;

    createMeshFromGrid_secondPass(scene_info, grid, vertex_is_active_buffer, (float3*) owlBufferGetPointer(mesh.vertices, 0), (int3*) owlBufferGetPointer(mesh.indices, 0));
    return mesh.n_triangles;
}

void writeMeshToObj(TriangleMeshOwl& mesh, fs::path data_dir, const char filename[])
{
    std::vector<float3> vertices(mesh.total_n_vertices);
    cudaMemcpy(vertices.data(), owlBufferGetPointer(mesh.vertices, 0), sizeof(float3) * mesh.total_n_vertices, cudaMemcpyDeviceToHost);

    std::vector<int3> indices(mesh.n_triangles);
    cudaMemcpy(indices.data(), owlBufferGetPointer(mesh.indices, 0), sizeof(int3) * mesh.n_triangles, cudaMemcpyDeviceToHost);
    
    writeMeshToObj(vertices, indices, mesh.n_triangles, data_dir, filename);
}
#endif