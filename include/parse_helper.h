
#pragma once

#include "common.h"

#include <filesystem>
#include <string>

#include "util/helper_math_extension.h"

void parse(CameraMatrix& matrix, const json& data)
{
    auto tmp = data.get<std::vector<std::vector<float>>>();

    matrix.m0 = make_float4(tmp[0][0], tmp[0][1], tmp[0][2], tmp[0][3]);
    matrix.m1 = make_float4(tmp[1][0], tmp[1][1], tmp[1][2], tmp[1][3]);
    matrix.m2 = make_float4(tmp[2][0], tmp[2][1], tmp[2][2], tmp[2][3]);
}


void parse(SceneInfo& info, const json &data)
{
    info.grid_resolution = data["grid_resolution"].get<int>();
    info.grid_nlvl = data["grid_nlvl"].get<int>();
    info.vals_per_lvl = info.grid_resolution * info.grid_resolution * info.grid_resolution;
    info.normalized = false;

    auto aabb_temp = data["aabb"].get<std::vector<float>>();
    assert(aabb_temp.size() == 6);
    info.aabb_from = make_float3(aabb_temp.at(0), aabb_temp.at(1), aabb_temp.at(2));
    info.aabb_to = make_float3(aabb_temp.at(3), aabb_temp.at(4), aabb_temp.at(5));

    if (data.contains("contraction") && data["contraction"].contains("aabb"))
    {
        info.contraction_type = ContractionType::WARP_AABB_LINF;
        
        aabb_temp = data["contraction"]["aabb"].get<std::vector<float>>();
        assert(aabb_temp.size() == 6);
        info.contraction_aabb_from = make_float3(aabb_temp.at(0), aabb_temp.at(1), aabb_temp.at(2));
        info.contraction_aabb_to = make_float3(aabb_temp.at(3), aabb_temp.at(4), aabb_temp.at(5));
    }
    else
    {
        info.contraction_type = ContractionType::AABB;
    }
    

    info.is_open_gl = data["is_open_gl"].get<bool>();
    info.alpha_thre = data["alpha_thre"].get<float>();
};


void parse(StepsizeInfo& info, const json &data)
{
    info.cone_angle = data["cone_angle"].get<float>();
    info.stepsize = data["stepsize"].get<float>();
    info.near = data["near"].get<float>();
    info.far = data["far"].get<float>();
};


void parse(CameraInfo& info, const json &data)
{
    int width = data["width"].get<int>();
    int height = data["height"].get<int>();
    info.resolution = make_int2(width, height);
    info.is_open_gl = data["is_open_gl"].get<bool>();

    auto intrinsics = data["intrinsics"].get<std::vector<std::vector<float>>>();
    info.focal = make_float2(intrinsics[0][0], intrinsics[1][1]);
    info.principal = make_float2(intrinsics[0][2], intrinsics[1][2]);
};

void parseFromNgp(CameraInfo& info, const json &data)
{
    // Use nerf synthetic width/height as default
    int width = data.value<int>("w", 800);
    int height = data.value<int>("h", 800);
    info.resolution = make_int2(width, height);
    info.is_open_gl = false;

    if (data.contains("fl_x")) // Output from ingp "colmap2nerf.py"
    {
        info.focal = make_float2(data["fl_x"].get<float>(), data["fl_y"].get<float>());
        info.principal = make_float2(data["cx"].get<float>(), data["cy"].get<float>());
    }
    else if (data.contains("camera_angle_x")) // NeRF synthetic dataset
    {
        info.focal = make_float2(0.5f * (float) width / tanf(0.5f * data["camera_angle_x"].get<float>()));
        info.principal = make_float2(info.resolution) * 0.5f;
    }
    else
    {
        throw std::runtime_error("Unknown Camera Info file source");      
    }
};


void parse(CameraPathDataEntry& entry, const json &data)
{
    parse(entry.transform_matrix, data["transform_matrix"]);

    entry.image_path = std::filesystem::path(data["image_path"].get<std::string>());
    if (!entry.image_path.has_extension()) entry.image_path.replace_extension(".png");
}

void parseFromNgp(CameraPathDataEntry& entry, const json &data)
{
    parse(entry.transform_matrix, data["transform_matrix"]);

    entry.image_path = std::filesystem::path(data["file_path"].get<std::string>());
    if (!entry.image_path.has_extension()) entry.image_path.replace_extension(".png");
}