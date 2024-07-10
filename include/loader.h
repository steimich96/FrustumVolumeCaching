/*
 * Copyright (C) 2024, Michael Steiner, Graz University of Technology.
 * This code is licensed under the MIT license.
 */

#pragma once

#include "common.h"
#include "nerf_network.h"
#include "util/buffer.h"

#include <filesystem>
#include <string>
namespace fs = std::filesystem;

enum DataSource
{
    INGP = 0,
    NERFACC = 1
};

inline std::string dataSourceSuffix(DataSource d)
{
    switch (d)
    {
    case DataSource::INGP:    return "ingp";
    case DataSource::NERFACC: return "nerfacc";
    default:                  return "unknown";
    }
}

struct LoaderData
{
    DataSource data_source;
    SceneInfo scene_info;
    StepsizeInfo stepsize_info;

    CudaBuffer<half> params_pos_enc;
    CudaBuffer<half> params_mlp_base;
    CudaBuffer<half> params_mlp_first_head;
    CudaBuffer<half> params_mlp_head;

    NerfNetwork<half> nerf_network;
    CudaBuffer<uint8_t> occupancy_grid_linear_uchar;
    CudaBuffer<uint8_t> occupancy_grid_linear_bitfield;
    CudaBuffer<uint8_t> occupancy_grid_morton_bitfield;
};

bool load_data(fs::path data_dir, LoaderData& loader_data);
bool load_camerapath(fs::path camera_path_filename, DataSource data_source, CameraInfo& camera_info, std::vector<CameraPathDataEntry>& camera_path);
bool load_ngp_camerapath(fs::path camera_path_filename, CameraInfo& camera_info, std::vector<CameraPathDataEntry>& camera_path);