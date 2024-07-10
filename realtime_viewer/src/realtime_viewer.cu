/*
 * Copyright (C) 2024, Michael Steiner, Graz University of Technology.
 * This code is licensed under the MIT license.
 */

#include "application.h"

#include "loader.h"

#include <cuda.h>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <cmath>

#include <argparse/argparse.hpp>
#include <json/json.hpp>

int main(int argc, char const *argv[])
{
    std::cout << "Initializing CUDA..." << std::endl;
    CUDA_CHECK_THROW(cudaFree(0));
    cudaPrintMemInfo();

    argparse::ArgumentParser program("realtime-viewer");

    program.add_argument("data")
        .help("either the path to the nerfacc export directory or to an ingp snapshot file");
    program.add_argument("-r", "--resolution")
        .default_value(std::vector<int>{1280, 720})
        .nargs(2)
        .help("Initial resolution when launchen (also fixed resolution for cache)")
        .scan<'d', int>();
    program.add_argument("-f", "--field-of-view")
        .default_value(60.f)
        .help("Initial FOV")
        .scan<'g', float>();
    program.add_argument("-s", "--single-buffering")
        .default_value(false)
        .implicit_value(true)
        .help("Use single cache buffer instead of double buffering (reduces memory requirements)");

    argparse::ArgumentParser render_command("render");
    render_command.add_argument("camera-path-file")
        .help("full path to camera path file for rendering");
    render_command.add_argument("-o", "--output-file")
        .help("full path to output file (.mp4) [ALL PNGs IN THIS DIRECTORY WILL BE DELETED]. Default: ./render_output/rendering.mp4")
        .default_value((std::filesystem::current_path() / "render_output/rendering.mp4").c_str());    

    program.add_subparser(render_command);

    try
    {
        program.parse_args(argc, argv);
    }
    catch (const std::runtime_error& err)
    {
        std::cout << "Argument parsing failed!" << std::endl;
        std::cout << err.what() << std::endl;
        std::cout << program;
        return 0;
    }

    // CameraInfo camera_info;
    // std::vector<CameraPathDataEntry> camera_path_data;
    // if (!load_camerapath(program.get<std::string>("camera-path-file"), DataSource::NERFACC, camera_info, camera_path_data) || camera_path_data.size() == 0)
    // {
    //     std::cout << "Could not load the camera path file" << std::endl;
    //     return 0;
    // }

    CameraMatrix initial_pose = {
        {0.875685, -0.145455, 0.460456, -0.499566},
        {0.0129797, 0.9603, 0.278666, -0.178782},
        {-0.48271, -0.238047, 0.842808, -0.854686},
    };

    auto res_vec = program.get<std::vector<int>>("--resolution");
    int2 resolution = {res_vec[0], res_vec[1]};
    float fov_deg = program.get<float>("--field-of-view");

    CameraPath cam_path;
    bool rendering = false;

    if (program.is_subcommand_used(render_command))
    {
        rendering = true;
        auto cam_path_file = render_command.get<std::string>("camera-path-file");
        if (!cam_path.load(cam_path_file))
        {
            std::cout << "Could not find camera path file: " << cam_path_file << std::endl;
            return 0;
        }

        std::filesystem::path output_filepath = render_command.get<std::string>("-o");
        if(!std::filesystem::exists(output_filepath.parent_path()))
        {
            std::cout << "Path to " << output_filepath.string() << " does not exist!" << std::endl;
            return 0;
        }
        cam_path.settings.filename = output_filepath.string();
        resolution = cam_path.settings.resolution;
        fov_deg = cam_path.settings.fov_x_deg;
    }

    float focal_x = ((float)resolution.x * 0.5f) / std::tan(fov_deg * 0.5f * (float)M_PI / 180.f);
    CameraInfo camera_info = {
        .resolution = resolution,
        .is_open_gl = false,
        .focal = {focal_x, focal_x},
        .principal = {(float)resolution.x / 2.f, (float)resolution.y / 2.f},
        .aperature = 0.0,  
        .focus_z = 0.7
    };

    Application realtime_viewer(resolution, program.get<std::string>("data"), {
            .interpol_variant = DensityInterpolVariant::Density,
            .interpol_function = InterpolFunction::Linear,
            .use_hw_interpol = true,
            .reweight_intermediates = true,
            .disable_reweighting_first_intermediate = true,
            .use_inter_brick_interpolation = true
        }, camera_info, initial_pose, fov_deg, cam_path, program.is_subcommand_used(render_command) ? false : (program["--single-buffering"] != true));

    if (rendering)
        realtime_viewer.renderCameraPath();
    else
        realtime_viewer.run();

    std::cout << std::endl << "Done!" << std::endl;
    return 0;
}