/*
 * Copyright (C) 2024, Michael Steiner, Graz University of Technology.
 * This code is licensed under the MIT license.
 */

#include "loader.h"

#include "parse_helper.h"

#include <cassert>
#include <exception>
#include <iostream>
#include <fstream>

#include <zstr.hpp>

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/reduce_sum.h>

__global__ void occupancy_grid_to_morton_kernel(int n_elements,
    SceneInfo scene_info,
    const uint8_t* __restrict__ occupancy_grid,
    uint8_t* __restrict__ occupancy_grid_morton)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements)
        return;

    int level = i / scene_info.vals_per_lvl;
    int intra_level_idx = i % scene_info.vals_per_lvl;

	uint32_t x = tcnn::morton3D_invert(intra_level_idx>>0);
	uint32_t y = tcnn::morton3D_invert(intra_level_idx>>1);
	uint32_t z = tcnn::morton3D_invert(intra_level_idx>>2);
    int occupancy_grid_idx = level * scene_info.vals_per_lvl + x * scene_info.grid_resolution * scene_info.grid_resolution + y * scene_info.grid_resolution + z;

    occupancy_grid_morton[i] = occupancy_grid[occupancy_grid_idx];
}

__global__ void occupancy_grid_uchar_to_bitfield_kernel(int n_elements,
    SceneInfo scene_info,
    const uint8_t* __restrict__ occupancy_grid_uchar,
    uint8_t* __restrict__ occupancy_grid_bitfield)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements)
        return;

	uint8_t bits = 0;
	for (uint8_t j = 0; j < 8; ++j) {
		bits |= occupancy_grid_uchar[i*8+j] > 0 ? ((uint8_t)1 << j) : 0;
	}

	occupancy_grid_bitfield[i] = bits;
}

__global__ void occupancy_grid_linear_to_bitfield_kernel(int n_elements,
    SceneInfo scene_info,
    const uint8_t* __restrict__ occupancy_grid_linear,
    uint8_t* __restrict__ occupancy_grid_linear_bitfield)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements)
        return;

	uint8_t bits = 0;
	for (uint8_t j = 0; j < 8; ++j) {
		bits |= occupancy_grid_linear[i*8+j] > 0 ? ((uint8_t)1 << j) : 0;
	}

	occupancy_grid_linear_bitfield[i] = bits;
}

__global__ void ingp_density_grid_to_occupancy_grid_kernel(int n_elements,
    SceneInfo scene_info,
    const float* __restrict__ density_grid,
    uint8_t* __restrict__ occupancy_grid,
    const float* __restrict__ mean_density_ptr)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements)
        return;

    int level = i / scene_info.vals_per_lvl;
    int intra_level_idx = i % scene_info.vals_per_lvl;
    
	uint32_t x = tcnn::morton3D_invert(intra_level_idx>>0);
	uint32_t y = tcnn::morton3D_invert(intra_level_idx>>1);
	uint32_t z = tcnn::morton3D_invert(intra_level_idx>>2);
    int occupancy_grid_idx = level * scene_info.vals_per_lvl + x * pow(scene_info.grid_resolution, 2) + y * scene_info.grid_resolution + z;

	float thresh = std::min(0.01f, *mean_density_ptr);
    occupancy_grid[occupancy_grid_idx] = density_grid[i] > thresh ? 1 : 0;
}

void ingp_density_grid_to_occupancy_grid(SceneInfo scene_info, CudaBuffer<half>& density_grid_hp, CudaBuffer<uint8_t>& occupancy_grid)
{
    int n_elements = scene_info.vals_per_lvl;
    int total_n_elements = n_elements * scene_info.grid_nlvl;

    CudaBuffer<float> density_grid_fp;
    density_grid_fp.resize(density_grid_hp.size());

    tcnn::parallel_for_gpu(density_grid_hp.size(), [density_grid=density_grid_fp.data(), density_grid_hp=density_grid_hp.data()] __device__ (size_t i) {
        density_grid[i] = (float)density_grid_hp[i];
    });

    CudaBuffer<float> density_grid_mean(tcnn::reduce_sum_workspace_size(n_elements));
    density_grid_mean.memset(0);

    tcnn::reduce_sum(density_grid_fp.data(), [n_elements] __device__ (float val) { return fmaxf(val, 0.f) / (n_elements); }, 
        density_grid_mean.data(), n_elements, cudaStreamDefault);
    
    occupancy_grid.resize(total_n_elements);

    ingp_density_grid_to_occupancy_grid_kernel<<<divRoundUp(total_n_elements, 128), 128>>>(total_n_elements, scene_info, density_grid_fp.data(),
        occupancy_grid.data(), density_grid_mean.data());
}

void occupancy_grid_uchar_to_bitfield(SceneInfo scene_info, CudaBuffer<uint8_t>& occupancy_grid_uchar, CudaBuffer<uint8_t>& occupancy_grid_bitfield)
{
    int n_elements = scene_info.vals_per_lvl;
    int total_n_elements = n_elements * scene_info.grid_nlvl;

    occupancy_grid_bitfield.resize(total_n_elements / 8);
    occupancy_grid_uchar_to_bitfield_kernel<<<divRoundUp(total_n_elements / 8, 128), 128>>>(total_n_elements / 8, scene_info,
                                                                                            occupancy_grid_uchar.data(), 
                                                                                            occupancy_grid_bitfield.data());
    CUDA_SYNC_CHECK_THROW();
}

void occupancy_grid_to_morton_bitfield(SceneInfo scene_info, CudaBuffer<uint8_t>& occupancy_grid_linear_uchar, CudaBuffer<uint8_t>& occupancy_grid_morton_bitfield)
{
    int n_elements = scene_info.vals_per_lvl;
    int total_n_elements = n_elements * scene_info.grid_nlvl;

    CudaBuffer<uint8_t> occupancy_grid_morton_uchar;
    occupancy_grid_morton_uchar.resize(total_n_elements);
    occupancy_grid_to_morton_kernel<<<divRoundUp(total_n_elements, 128), 128>>>(total_n_elements, scene_info,
                                                                                occupancy_grid_linear_uchar.data(), occupancy_grid_morton_uchar.data());
    CUDA_SYNC_CHECK_THROW();

    occupancy_grid_uchar_to_bitfield(scene_info, occupancy_grid_morton_uchar, occupancy_grid_morton_bitfield);
}

bool load_ingp_snapshot(fs::path snapshot_filepath, LoaderData& loader_data)
{
    std::ifstream snapshot_file{snapshot_filepath, std::ios::in | std::ios::binary};
    if (!snapshot_file.is_open())
    {
        std::cout << "Could not load snapshot file!" << std::endl;
        return false;
    }

    zstr::istream zf{snapshot_file};
    json config = json::from_msgpack(zf);

	if (!config.contains("snapshot"))
    {
        std::cout << "File '" << snapshot_filepath.c_str() << "' does not contain a snapshot." << std::endl;
        return false;
    }

	const auto& snapshot = config["snapshot"];
	if (!snapshot.contains("mode") || snapshot["mode"].get<std::string>() != std::string("nerf"))
    {
        std::cout << "File '" << snapshot_filepath.c_str() << "' does not contain a nerf snapshot." << std::endl;
        return false;
    }


    std::cout << "Loading SceneInfo and StepsizeInfo..." << std::endl;
    int aabb_scale = snapshot["nerf"]["aabb_scale"].get<int>();
    loader_data.scene_info.grid_resolution = 128;
    loader_data.scene_info.grid_nlvl = log2(aabb_scale) + 1;
    loader_data.scene_info.vals_per_lvl = pow(128, 3);
    loader_data.scene_info.normalized = true;

    std::vector<float> aabb_min = snapshot["aabb"]["min"].get<std::vector<float>>();
    std::vector<float> aabb_max = snapshot["aabb"]["max"].get<std::vector<float>>();
    loader_data.scene_info.aabb_from = make_float3(aabb_min.at(0), aabb_min.at(1), aabb_min.at(2));
    loader_data.scene_info.aabb_to = make_float3(aabb_max.at(0), aabb_max.at(1), aabb_max.at(2));

    loader_data.scene_info.contraction_type = ContractionType::AABB;

    loader_data.scene_info.is_open_gl = false;
    loader_data.scene_info.alpha_thre = 0.0f;


    loader_data.stepsize_info.cone_angle = loader_data.scene_info.grid_nlvl <= 1 ? 0.0f : 1.0f / 256.0f;
    loader_data.stepsize_info.stepsize = sqrt(3.0f) / 1024.f;
    loader_data.stepsize_info.near = 0.0f;
    loader_data.stepsize_info.far = 1e10f;

    loader_data.stepsize_info.setAdditionalInfo(loader_data.scene_info);

    json mlp_base_config;
    mlp_base_config["encoding"] = config["encoding"];
    int grid_base_resolution = mlp_base_config["encoding"]["base_resolution"].get<int>();
    int grid_n_levels = mlp_base_config["encoding"]["n_levels"].get<int>();

    mlp_base_config["encoding"]["per_level_scale"] = exp(log(2048.f * aabb_scale / (float) grid_base_resolution) / (float) (grid_n_levels - 1.f));
    mlp_base_config["network"] = config["network"];
    mlp_base_config["n_output"] = 16;

    json mlp_head_config;
    mlp_head_config["encoding"] = config["dir_encoding"];
    mlp_head_config["network"] = config["rgb_network"];

    std::cout << "Loading Network Parameters..." << std::endl;
    nlohmann::json::binary_t params_cpu = snapshot["params_binary"];
    loader_data.params_mlp_base.resize(params_cpu.size() / sizeof(half));
    CUDA_CHECK_THROW(cudaMemcpy(loader_data.params_mlp_base.data(), params_cpu.data(), loader_data.params_mlp_base.sizeInBytes(), cudaMemcpyHostToDevice));

    std::cout << "Initializing Network..." << std::endl;
    loader_data.nerf_network.init(mlp_base_config, mlp_head_config);
    loader_data.nerf_network.set_params(loader_data.params_mlp_base.data());

    std::cout << "Loading Occupancy Grid..." << std::endl;
    CudaBuffer<half> density_grid_hp;
    nlohmann::json::binary_t density_grid_hp_cpu = snapshot["density_grid_binary"];
    density_grid_hp.resize(density_grid_hp_cpu.size() / sizeof(half));
    CUDA_CHECK_THROW(cudaMemcpy(density_grid_hp.data(), density_grid_hp_cpu.data(), density_grid_hp.sizeInBytes(), cudaMemcpyHostToDevice));

    ingp_density_grid_to_occupancy_grid(loader_data.scene_info, density_grid_hp, loader_data.occupancy_grid_linear_uchar);
    occupancy_grid_uchar_to_bitfield(loader_data.scene_info, loader_data.occupancy_grid_linear_uchar, loader_data.occupancy_grid_linear_bitfield);
    occupancy_grid_to_morton_bitfield(loader_data.scene_info, loader_data.occupancy_grid_linear_uchar, loader_data.occupancy_grid_morton_bitfield);
    
    return true;
}

bool load_nerfacc_export(fs::path data_dir, LoaderData& loader_data)
{
    std::ifstream config_file(data_dir / "config.json");
    if (!config_file.is_open())
    {
        std::cout << "Could not load config file!" << std::endl;
        return false;
    }
    json config_json_data = json::parse(config_file);

    std::cout << "Loading SceneInfo and StepsizeInfo..." << std::endl;
    parse(loader_data.scene_info, config_json_data["scene"]);
    parse(loader_data.stepsize_info, config_json_data["scene"]);
    loader_data.stepsize_info.setAdditionalInfo(loader_data.scene_info);

    std::cout << "Loading Network Parameters..." << std::endl;
    bool pos_enc_separate_file = loader_data.params_pos_enc.readFileAndUpload(data_dir, "pos_enc.dat");
    loader_data.params_mlp_base.readFileAndUpload(data_dir, "mlp_base.dat");
    loader_data.params_mlp_head.readFileAndUpload(data_dir, "mlp_head.dat");
    if (config_json_data.contains("mlp_first_head")) loader_data.params_mlp_first_head.readFileAndUpload(data_dir, "mlp_first_head.dat");

    std::cout << "Initializing Network..." << std::endl;
    if (config_json_data.contains("mlp_first_head"))
    {
        loader_data.nerf_network.init_viewdep(config_json_data["mlp_base"], config_json_data["mlp_first_head"], config_json_data["mlp_head"]);
        
        std::cout << "N Params - Loader:  " << loader_data.params_pos_enc.size() << ", "  << loader_data.params_mlp_base.size() << ", " 
                  << loader_data.params_mlp_first_head.size() << ", " << loader_data.params_mlp_head.size() << std::endl;

        loader_data.nerf_network.set_params(loader_data.params_pos_enc.data(), loader_data.params_mlp_base.data(),
                                            loader_data.params_mlp_first_head.data(), loader_data.params_mlp_head.data());
    }
    else
    {
        loader_data.nerf_network.init(config_json_data["mlp_base"], config_json_data["mlp_head"]);

        std::cout << "N Params - Loader:  " << loader_data.params_pos_enc.size() << ", "  << loader_data.params_mlp_base.size() << ", " 
                  << loader_data.params_mlp_head.size() << std::endl;

        if (pos_enc_separate_file)
            loader_data.nerf_network.set_params(loader_data.params_pos_enc.data(), loader_data.params_mlp_base.data(), loader_data.params_mlp_head.data());
        else
            loader_data.nerf_network.set_params(loader_data.params_mlp_base.data(), loader_data.params_mlp_head.data());
    }

    std::cout << "Loading Occupancy Grid..." << std::endl;
    loader_data.occupancy_grid_linear_uchar.readFileAndUpload(data_dir, "occupancy_grid.dat");

    assert(loader_data.occupancy_grid_linear_uchar.size() == (loader_data.scene_info.vals_per_lvl * loader_data.scene_info.grid_nlvl));
    occupancy_grid_uchar_to_bitfield(loader_data.scene_info, loader_data.occupancy_grid_linear_uchar, loader_data.occupancy_grid_linear_bitfield);
    occupancy_grid_to_morton_bitfield(loader_data.scene_info, loader_data.occupancy_grid_linear_uchar, loader_data.occupancy_grid_morton_bitfield);
    
    return true;
}

bool load_data(fs::path data_dir, LoaderData& loader_data)
{
    if (data_dir.extension() == fs::path(".ingp"))
    {
        std::cout << "Loading Data from ingp snapshot..." << std::endl;
        loader_data.data_source = DataSource::INGP;
        return load_ingp_snapshot(data_dir, loader_data);
    }    
    else
    {
        std::cout << "Loading Data from Nerfacc Export..." << std::endl;
        loader_data.data_source = DataSource::NERFACC;
        return load_nerfacc_export(data_dir, loader_data);
    }
}

CameraMatrix nerf_matrix_to_ngp(CameraMatrix& nerf_matrix) {
    CameraMatrix result = nerf_matrix;

    float4 scaling {1.f, -1.f, -1.f, 0.33f};
    float4 offset {0.f, 0.f, 0.f, 0.5f};

    result.m0 = result.m0 * scaling + offset;
    result.m1 = result.m1 * scaling + offset;
    result.m2 = result.m2 * scaling + offset;

    // Cycle axes xyz <- yzx
    float4 tmp = result.m0;
    result.m0 = result.m1;
    result.m1 = result.m2;
    result.m2 = tmp;

    return result;
}

bool load_nerfacc_camerapath(fs::path camera_path_filename, CameraInfo& camera_info, std::vector<CameraPathDataEntry>& camera_path)
{

    std::ifstream camera_path_file(camera_path_filename);
    if (!camera_path_file.is_open())
    {
        std::cout << "Could not load camera path file!" << std::endl;
        return false;
    }
    json camera_path_json_data = json::parse(camera_path_file);

    parse(camera_info, camera_path_json_data);
    for (auto &frame_json_data : camera_path_json_data["frames"])
    {
        CameraPathDataEntry entry;
        parse(entry, frame_json_data);
        camera_path.push_back(entry);
    }
    std::cout << "Loaded " << camera_path.size() << " camera path frames" << std::endl;

    return true;
}

bool load_ngp_camerapath(fs::path camera_path_filename, CameraInfo& camera_info, std::vector<CameraPathDataEntry>& camera_path)
{
    std::cout << "Loading ingp Camera Path File..." << std::endl;

    std::ifstream camera_path_file(camera_path_filename);
    if (!camera_path_file.is_open())
    {
        std::cout << "Could not load camera path file!" << std::endl;
        return false;
    }
    json camera_path_json_data = json::parse(camera_path_file);

    parseFromNgp(camera_info, camera_path_json_data);
    for (auto &frame_json_data : camera_path_json_data["frames"])
    {
        CameraPathDataEntry entry;
        parseFromNgp(entry, frame_json_data);
        entry.transform_matrix = nerf_matrix_to_ngp(entry.transform_matrix);
        camera_path.push_back(entry);
    }
    std::cout << "Loaded " << camera_path.size() << " camera path frames" << std::endl;

    return true;
}

bool load_camerapath(fs::path camera_path_filename, DataSource data_source, CameraInfo& camera_info, std::vector<CameraPathDataEntry>& camera_path)
{
    switch (data_source)
    {
    case DataSource::INGP:
        return load_ngp_camerapath(camera_path_filename, camera_info, camera_path);
    
    case DataSource::NERFACC:
        return load_nerfacc_camerapath(camera_path_filename, camera_info, camera_path);

    default:
        throw std::runtime_error("Unknown Data Source!");
    }
}
