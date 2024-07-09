
#include "cache.h"
#include "cache_performance_renderer.h"
#include "common.h"
#include "loader.h"
#include "nerf_network.h"
#include "performance_renderer.h"
#include "raymarch_common.h"
#include "util/buffer.h"
#include "util/debug_buffer.h"
#include "util/cuda_helper.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image/stb_image.h>
#include <stb_image/stb_image_write.h>

#include <argparse/argparse.hpp>
#include <json/json.hpp>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/rotate_vector.hpp>

#include <iostream>
#include <sstream>
#include <fstream>

using json = nlohmann::json;

struct PerformanceMetrics {
    std::string image_name;
    int n_bricks_set;
    int n_bricks_reserved;
    bool overflow;
    uint64_t n_cache_samples;
    uint64_t n_resamples;
};

struct TimingMetrics {
    std::string image_name;
    float cache_init_ms;
    float render_time_ms;
    std::vector<TimerEntry> detailed_timings;
};

void to_json(nlohmann::json& j, const PerformanceMetrics& p) 
{
	j = nlohmann::json{
		{"image", p.image_name},
		{"n_bricks_set", p.n_bricks_set},
		{"n_bricks_reserved", p.n_bricks_reserved},
		{"overflow", p.overflow},
		{"n_cache_samples", p.n_cache_samples},
		{"n_resamples", p.n_resamples},
	};
}

void to_json(nlohmann::json& j, const TimingMetrics& t) 
{
	j = nlohmann::json{
		{"image", t.image_name},
        {"cache_init_ms", t.cache_init_ms},
        {"render_time_ms", t.render_time_ms},
		{"timings", t.detailed_timings}
	};
}

static CameraMatrix lookAt(glm::vec3 eye, glm::vec3 center, glm::vec3 up)
{
    glm::vec3 const view_dir_new(normalize(center - eye));
    glm::vec3 const right_new(normalize(cross(up, view_dir_new)));
    glm::vec3 const up_new(cross(view_dir_new, right_new));

    return {
        {right_new.x, up_new.x, view_dir_new.x, eye.x},
        {right_new.y, up_new.y, view_dir_new.y, eye.y},
        {right_new.z, up_new.z, view_dir_new.z, eye.z},
    };
}

int main(int argc, char const *argv[])
{
    argparse::ArgumentParser program("eval");

    program.add_argument("data").help("either the path to the nerfacc export directory or to an ingp snapshot file");
    program.add_argument("camera-transforms").help("path to a .json file containing the camera transforms");
    program.add_argument("output-dir").help("path to the output directory");

    program.add_argument("-sc", "--sanity-check").help("perform sanity PSNR checks against ref image").default_value(false).implicit_value(true);
    program.add_argument("-cp", "--copy-original-images").help("copies the original image to the output directory").default_value(false).implicit_value(true);
    program.add_argument("-w", "--write-images").help("write images to output directory").default_value(false).implicit_value(true);
    program.add_argument("-wi", "--write-cache-init-images").help("write images of cache init position").default_value(false).implicit_value(true);
    program.add_argument("-wc", "--write-contrib-images").help("write images of cache/resample contribution").default_value(false).implicit_value(true);

    program.add_argument("-wc", "--write-contrib-images").help("write images of cache/resample contribution").default_value(false).implicit_value(true);

    program.add_argument("-or", "--overwrite-resolution").help("Do not use resolution of input images, but custom resolution (certain things might not work anymore)").default_value(false).implicit_value(true);
    program.add_argument("-r", "--resolution").help("Initial resolution when launchen (also fixed resolution for cache)").default_value(std::vector<int>{1280, 720}).nargs(2).scan<'d', int>();

    program.add_argument("-nrit", "--num-render-iterations")
        .help("Number of times the render functions should be called (for timing puroses)")
        .default_value(1U)
        .scan<'u', unsigned int>();

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

    auto data_path = program.get<std::string>("data");
    auto camera_transforms_path = program.get<std::string>("camera-transforms");
    auto output_dir = program.get<std::string>("output-dir");

    fs::create_directories(output_dir);

    bool sanity_check = program.get<bool>("--sanity-check");
    bool copy_original_images = program.get<bool>("--copy-original-images");

    bool write_images = program.get<bool>("--write-images");
    bool write_cache_init_images = write_images && program.get<bool>("--write-cache-init-images");
    bool write_contrib_images = write_images && program.get<bool>("--write-contrib-images");

    int num_render_iterations = program.get<unsigned int>("--num-render-iterations");

    LoaderData loader;

    if(!load_data(data_path, loader))
        return 1;

    SceneInfo scene_info = loader.scene_info;
    StepsizeInfo dataset_stepsize_info = loader.stepsize_info;

    CameraInfo dataset_camera_info;
    std::vector<CameraPathDataEntry> camera_path_data;

    if (!load_camerapath(camera_transforms_path, loader.data_source, dataset_camera_info, camera_path_data))
        return 1;

    RaymarchInfo dataset_rm_info{
            cam_info : dataset_camera_info,
            next_cam_info : {},
            stepsize_info : dataset_stepsize_info,
            sample_index : 0,
            motion_blur : false,
            deterministic : true
    };

    bool overwrite_resolution = program.get<bool>("--overwrite-resolution");
    std::vector<int> arg_resolution = program.get<std::vector<int>>("--resolution");
    int2 custom_resolution = {arg_resolution[0], arg_resolution[1]};

    StepsizeInfo cache_init_stepsize_info = dataset_stepsize_info;
    CameraInfo cache_camera_info = dataset_camera_info;

    StepsizeInfo render_stepsize_info = dataset_stepsize_info;
    CameraInfo render_camera_info = dataset_camera_info;
    
    if (overwrite_resolution)
    {
        render_camera_info.resize(custom_resolution);
        cache_camera_info.resize(custom_resolution);
    }

    RaymarchInfo cache_rm_info{
            cam_info : cache_camera_info,
            next_cam_info : {},
            stepsize_info : cache_init_stepsize_info,
            sample_index : 0,
            motion_blur : false,
            deterministic : true
    };
    RaymarchInfo render_rm_info{
            cam_info : render_camera_info,
            next_cam_info : {},
            stepsize_info : render_stepsize_info,
            sample_index : 0,
            motion_blur : false,
            deterministic : true
    };

    ImageBuffer sanity_check_image_buffer_1(dataset_camera_info.resolution);
    ImageBuffer sanity_check_image_buffer_2(dataset_camera_info.resolution);


    ImageBuffer render_image_buffer(render_camera_info.resolution);
    ImageBuffer render_debug_buffer(render_camera_info.resolution);

    DebugData debug_data;
    DebugBuffer<float> cache_render_contrib_resample(render_camera_info.resolution);
    DebugBuffer<float> cache_render_contrib_cache(render_camera_info.resolution);
    debug_data.buffers.push_back(&cache_render_contrib_resample);
    debug_data.buffers.push_back(&cache_render_contrib_cache);

    CudaBuffer<uint8_t>& occupancy_grid = loader.occupancy_grid_morton_bitfield;

    PerformanceRenderer renderer(scene_info, occupancy_grid, loader.nerf_network);
    renderer.resizeRenderbuffers(render_camera_info.resolution);

    PerformanceRenderer sanity_renderer(scene_info, occupancy_grid, loader.nerf_network);
    sanity_renderer.resizeRenderbuffers(dataset_camera_info.resolution);

    Cache cache;
    CacheSettings cache_settings{
        .interpol_variant = loader.nerf_network.m_first_latent_is_density ? DensityInterpolVariant::DensityIntermediates : DensityInterpolVariant::Density,
        .interpol_function = static_cast<InterpolFunction>(0),
        .use_hw_interpol = true,
        .reweight_intermediates = true,
        .disable_reweighting_first_intermediate = true,
        .use_inter_brick_interpolation = true
    };

    CachePerformanceRenderer cache_renderer(scene_info, occupancy_grid, loader.nerf_network, &cache, &cache);
    cache_renderer.resizeCache(cache_camera_info.resolution, cache_init_stepsize_info);
    cache_renderer.resizeRenderbuffers(cache_camera_info.resolution);
    cache_renderer.setCacheSettings(cache_settings);

    std::filesystem::create_directories(std::filesystem::absolute(output_dir));

    SimpleCudaTimer timer;
    float cache_init_ms = 0.f;
    float render_time_ms = 0.f;

    std::array<int, 3> view_degrees = { 5, 10, 15 };
    const int n_circle_samples = 6;

    float dt = dataset_rm_info.stepsize_info.stepsize;
    std::array<float, 6> translation_steps = { -0.25f, -0.1f, -dt * 0.5f, dt * 0.5f, 0.1f, 0.25f };

    int i = 0;
    for(auto& camera_path_entry : camera_path_data)
    {
        std::vector<PerformanceMetrics> perf_metrics;
        std::vector<TimingMetrics> perf_timings;

        std::cerr << std::to_string(i) << "/" << std::to_string(camera_path_data.size()) << "\r" << std::flush;

        std::string out_dir_name = std::string(output_dir) + "/" + camera_path_entry.image_path.stem().string();
        std::filesystem::create_directories(std::filesystem::absolute(out_dir_name));

        // Copy original jpg image
        if (copy_original_images)
            std::filesystem::copy(std::filesystem::path(camera_transforms_path).remove_filename() / camera_path_entry.image_path, out_dir_name + "/orig.jpg", std::filesystem::copy_options::skip_existing);



        auto& camera_matrix = camera_path_entry.transform_matrix;
        render_rm_info.cam_info.cam2world = camera_matrix;
        render_rm_info.cam_info.world2cam = camera_matrix.inverse();

        cache_rm_info.cam_info.cam2world = camera_matrix;
        cache_rm_info.cam_info.world2cam = camera_matrix.inverse();

        auto tmp_rm_info = dataset_rm_info;
        tmp_rm_info.cam_info.cam2world = camera_matrix;
        tmp_rm_info.cam_info.world2cam = camera_matrix.inverse();

        if (sanity_check)
        {
            sanity_check_image_buffer_1.readFromFile(std::filesystem::path(camera_transforms_path).remove_filename() / camera_path_entry.image_path);
            sanity_renderer.render(tmp_rm_info, sanity_check_image_buffer_2);

            float render_psnr = sanity_check_image_buffer_1.computePSNR(sanity_check_image_buffer_2, false, "");
            if (render_psnr < 15.f)
            {
                std::cerr << "SANITY CHECK FAILED: " << camera_path_entry.image_path.stem().string() << ", PSNR: " << render_psnr << std::endl;
                sanity_check_image_buffer_2.writeToFile(out_dir_name + "/sanity_check_error.png");
                break;
            }
        }


        // Render without cache
        renderer.render(render_rm_info, render_image_buffer); // warmup
        renderer._timer.reset();

        timer.start();
        for (int render_it = 0; render_it < num_render_iterations; render_it++)
            renderer.render(render_rm_info, render_image_buffer);
        float render_time_new_ms = timer.stopElapsed();


        nlohmann::json j_metrics = {};
        nlohmann::json j_timings = {};

        j_timings["no_cache"] = TimingMetrics {out_dir_name + "/no_cache", 0.0f,
                                               render_time_new_ms / num_render_iterations,
                                               renderer._timer.getTimings(num_render_iterations)};

        if (write_images)
            render_image_buffer.writeToFile(out_dir_name + "/no_cache.png");

        if (write_contrib_images)
        {
            cache_render_contrib_resample.set(0);
            cache_render_contrib_cache.set(0);
        }


        // Render with cache
        cache_renderer.initCacheBlockwise(cache_rm_info, render_debug_buffer);
        cache_renderer.render(render_rm_info, render_image_buffer, write_contrib_images ? &debug_data : nullptr);

        if (sanity_check)
        {
            renderer.render(render_rm_info, render_image_buffer);
            cache_renderer.render(render_rm_info, render_debug_buffer);

            float render_psnr = render_image_buffer.computePSNR(render_debug_buffer, false, "");
            if (render_psnr < 35.f)
            {
                std::cerr << "SANITY CHECK FAILED (CACHE RENDER): " << camera_path_entry.image_path.stem().string() << ", PSNR: " << render_psnr << std::endl;
                render_debug_buffer.writeToFile(out_dir_name + "/sanity_check_error_cache.png");
                break;
            }
        }

        if (write_images)
            render_image_buffer.writeToFile(out_dir_name + "/cache.png");

        // Calculate focus point by projecting vector from origin to camera onto view direction
        glm::vec3 view_dir = glm::normalize(glm::vec3{camera_matrix.m0.z, camera_matrix.m1.z, camera_matrix.m2.z});
        glm::vec3 up = glm::normalize(glm::vec3{camera_matrix.m0.y, camera_matrix.m1.y, camera_matrix.m2.y});
        glm::vec3 cam_pos = {camera_matrix.m0.w, camera_matrix.m1.w, camera_matrix.m2.w};
        auto proj = glm::dot(-cam_pos, view_dir);
        auto focus_point = cam_pos + proj * view_dir;
        auto cam_vector = cam_pos - focus_point;

        // Render looking at focus point
        RaymarchInfo focus_rm_info = render_rm_info;
        focus_rm_info.cam_info.cam2world = lookAt(cam_pos, focus_point, up);
        focus_rm_info.cam_info.world2cam = focus_rm_info.cam_info.cam2world.inverse();


        // Render rotations
        for (int deg : view_degrees)
        {
            std::string out_sub_dir_name = out_dir_name + "/rot/" + std::to_string(deg);

            if (write_images)
                std::filesystem::create_directories(std::filesystem::absolute(out_sub_dir_name));

            float phi = float(deg) * PI() / 180.f;
            glm::vec3 start_vector = glm::rotate(cam_vector, phi, up);

            for (int theta_i = 0; theta_i < n_circle_samples; theta_i++)
            {
                float theta = theta_i / (float) n_circle_samples * (2.0f * PI());

                CameraInfo subpose_cam_info = render_rm_info.cam_info;
                auto new_pos = glm::rotate(start_vector, theta, glm::normalize(cam_vector));

                subpose_cam_info.cam2world = lookAt(focus_point + new_pos, focus_point, up);
                subpose_cam_info.world2cam = subpose_cam_info.cam2world.inverse();

                RaymarchInfo subpose_rm_info = render_rm_info;
                subpose_rm_info.cam_info = subpose_cam_info;

                std::string image_name = std::to_string(int(round(theta * 180.f / PI())));
                if (write_cache_init_images)
                {
                    renderer.render(subpose_rm_info, render_image_buffer);
                    render_image_buffer.writeToFile(out_sub_dir_name + "/" + image_name + "_init.png");
                }                

                timer.start();
                cache_renderer.initCacheBlockwise(subpose_rm_info, render_debug_buffer);
                cache_init_ms = timer.stopElapsed();

                if (write_contrib_images)
                {
                    cache_render_contrib_resample.set(0);
                    cache_render_contrib_cache.set(0);
                }
                cache_renderer._timer.reset();

                // Rendering with Motion Blur and Depth of Field
                // RaymarchInfo tmp_rm_info = render_rm_info;
                // tmp_rm_info.next_cam_info = subpose_rm_info.cam_info;
                // tmp_rm_info.deterministic = false;
                // tmp_rm_info.cam_info.aperature = 0.01f;
                // tmp_rm_info.cam_info.focus_z = 1.2f;
                // tmp_rm_info.sample_index = 0;
                // tmp_rm_info.motion_blur = true;

                timer.start();
                for (int render_it = 0; render_it < num_render_iterations; render_it++)
                {
                    cache_renderer.render(render_rm_info, render_image_buffer, write_contrib_images ? &debug_data : nullptr);
                }
                render_time_ms = timer.stopElapsed();

                if (write_images)
                    render_image_buffer.writeToFile(out_sub_dir_name + "/" + image_name + ".png");

                if (write_contrib_images)
                {
                    cache_render_contrib_resample.writeToFile(out_sub_dir_name + "/" + image_name + "_contr_new.png", createMinMaxConverter<float>(0.0f, 1.0f));
                    cache_render_contrib_cache.writeToFile(out_sub_dir_name + "/" + image_name + "_contr_cache.png", createMinMaxConverter<float>(0.0f, 1.0f));
                }

                perf_metrics.push_back({std::to_string(deg) + "/" + image_name, 
                    cache_renderer._init_stats.n_bricks_set, 
                    cache_renderer._init_stats.n_bricks_reserved, 
                    cache_renderer._init_stats.n_bricks_set >= cache_renderer._init_stats.n_bricks_reserved,
                    cache_renderer._render_stats.n_cache_hits, 
                    cache_renderer._render_stats.n_resamples});

                perf_timings.push_back({"r_" + std::to_string(deg) + "/" + image_name, 
                    cache_init_ms, 
                    render_time_ms / num_render_iterations, 
                    cache_renderer._timer.getTimings(num_render_iterations)});

                //cache_renderer._timer.print();
            }
        }


        j_metrics["rot"] = perf_metrics;
        j_timings["rot"] = perf_timings;
        perf_metrics.clear();
        perf_timings.clear();


        std::string out_sub_dir_name = out_dir_name + "/trans";

        if (write_images)
            std::filesystem::create_directories(std::filesystem::absolute(out_sub_dir_name));

        // Render translations
        for (float step : translation_steps)
        {
            CameraInfo subpose_cam_info = render_rm_info.cam_info;

            auto new_pos = cam_pos + step * view_dir;
            subpose_cam_info.cam2world.m0.w = new_pos.x;
            subpose_cam_info.cam2world.m1.w = new_pos.y;
            subpose_cam_info.cam2world.m2.w = new_pos.z;
            subpose_cam_info.world2cam = subpose_cam_info.cam2world.inverse();

            RaymarchInfo subpose_rm_info = render_rm_info;
            subpose_rm_info.cam_info = subpose_cam_info;

            std::string image_name = std::to_string(step);
            if (write_cache_init_images)
            {
                renderer.render(subpose_rm_info, render_image_buffer);
                render_image_buffer.writeToFile(out_sub_dir_name + "/" + image_name + "_init.png");
            }

            timer.start();
            cache_renderer.initCacheBlockwise(subpose_rm_info, render_debug_buffer);
            cache_init_ms = timer.stopElapsed();

            if (write_contrib_images)
            {
                cache_render_contrib_resample.set(0);
                cache_render_contrib_cache.set(0);
            }
            cache_renderer._timer.reset();

            timer.start();
            for (int render_it = 0; render_it < num_render_iterations; render_it++)
                cache_renderer.render(render_rm_info, render_image_buffer, write_contrib_images ? &debug_data : nullptr);
            render_time_ms = timer.stopElapsed();

            if (write_images)
                render_image_buffer.writeToFile(out_sub_dir_name + "/" + image_name + ".png");

            if (write_contrib_images)
            {
                cache_render_contrib_resample.writeToFile(out_sub_dir_name + "/" + image_name + "_contr_new.png", createMinMaxConverter<float>(0.0f, 1.0f));
                cache_render_contrib_cache.writeToFile(out_sub_dir_name + "/" + image_name + "_contr_cache.png", createMinMaxConverter<float>(0.0f, 1.0f));
            }

            perf_metrics.push_back({image_name,
                                    cache_renderer._init_stats.n_bricks_set,
                                    cache_renderer._init_stats.n_bricks_reserved,
                                    cache_renderer._init_stats.n_bricks_set >= cache_renderer._init_stats.n_bricks_reserved,
                                    cache_renderer._render_stats.n_cache_hits,
                                    cache_renderer._render_stats.n_resamples});

            perf_timings.push_back({"t_" + image_name, cache_init_ms,
                                    render_time_ms / num_render_iterations,
                                    cache_renderer._timer.getTimings(num_render_iterations)});
        }

        j_metrics["trans"] = perf_metrics;
        j_timings["trans"] = perf_timings;
        j_timings["render_time_new_ms"] = render_time_new_ms / num_render_iterations;

        std::ofstream f_m(out_dir_name + "/metrics.json");
        f_m << j_metrics;

        std::ofstream f_t(out_dir_name + "/timings.json");
        f_t << j_timings;

        i++;
    }

    return 0;
}