/*
 * Copyright (C) 2024, Michael Steiner, Graz University of Technology.
 * This code is licensed under the MIT license.
 */

#include "application.h"

#include "core.h"
#include "event_dispatcher.h"
#include "window.h"
#include "util/helper_math.h"

#include <imgui.h>
#include <ImGuiFileDialog.h>

#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <numeric>
#include <chrono>
namespace fs = std::filesystem;

using json = nlohmann::json;

using ms = std::chrono::duration<float, std::milli>;

float mean(std::vector<float> v)
{
    return std::accumulate(v.begin(), v.end(), 0.0f) / v.size();
}

float min(std::vector<float> v)
{
    return *std::min_element(v.begin(), v.end());
}

float max(std::vector<float> v)
{
    return *std::max_element(v.begin(), v.end());
}

Application::Application(int2 resolution, std::filesystem::path data_dir, CacheSettings cache_settings, CameraInfo camera_info, CameraMatrix initial_pose, float fov_x, CameraPath cam_path, bool double_buffering)
    : m_window(BIND_EVENT_CALLBACK(Application::onEvent), WindowConfig{"NeRF Frustum Volume Caching Viewer", static_cast<uint32_t>(resolution.x), static_cast<uint32_t>(resolution.y), false})
    , m_interop_buffer(resolution)
    , m_debug_buffer(resolution)
    , m_cache_resolution(resolution)
    , m_running(false)
    , m_render_caching(true)
    , m_camera({0.f, 0.f, 0.f}, .8f, camera_info)
    , m_camera_path(cam_path)
    , m_fov_x(fov_x)
{
    MY_CHECK_THROW(load_data(data_dir, m_data_loader), "Failed to load data");

    m_scene_info = m_data_loader.scene_info;
    StepsizeInfo stepsize_info = m_data_loader.stepsize_info;

    int2 render_resolution = resolution;

    cudaPrintMemInfo();

    CudaBuffer<uint8_t>& occupancy_grid = GRID_IS_MORTON 
        ? m_data_loader.occupancy_grid_morton_bitfield
        : m_data_loader.occupancy_grid_linear_bitfield;

    cache_settings.interpol_variant = m_data_loader.nerf_network.m_first_latent_is_density
        ? DensityInterpolVariant::DensityIntermediates
        : DensityInterpolVariant::Density;

    StepsizeInfo cache_init_stepsize_info = stepsize_info;
    if (m_cache_double_sampling_rate)
        cache_init_stepsize_info.changeSamplingRate(2.0f, m_scene_info);

    m_cache_renderer = std::make_shared<CachePerformanceRenderer>(m_scene_info, occupancy_grid, m_data_loader.nerf_network, &m_cache, double_buffering ? &m_cache_next : &m_cache);
    m_cache_renderer->resizeCache(m_cache_resolution, cache_init_stepsize_info);
    m_cache_renderer->resizeRenderbuffers(m_cache_resolution);
    m_cache_renderer->setCacheSettings(cache_settings);
    
    m_renderer = std::make_shared<PerformanceRenderer>(m_scene_info, occupancy_grid, m_data_loader.nerf_network);
    m_renderer->resizeRenderbuffers(render_resolution);

    m_camera.setCameraMatrix(initial_pose);
    m_camera.resize(render_resolution);

    m_ray_march_info = RaymarchInfo { m_camera.getInfo(), {}, stepsize_info, 0, false, false};

    initCache();

    m_double_buffering = double_buffering;
    if (double_buffering)
        m_cache_update_thread = std::thread(&Application::updateCache, this);
}

Application::~Application()
{
    m_window.close();
}

void Application::destroy()
{
    m_update_cache_cv.notify_one();
    if (m_double_buffering)
        m_cache_update_thread.join();
    m_interop_buffer.destroy();
}

void Application::renderFrame()
{
    m_interop_buffer.map();
    m_window.beginFrame();

    if ((m_render_caching && !m_rendering_path) || (m_rendering_path && m_camera_path.settings.render_from_cache))
    {
        m_cache_renderer->render(m_ray_march_info, m_interop_buffer);
    }
    else
        m_renderer->render(m_ray_march_info, m_interop_buffer);

    // m_interop_buffer.swap();
    m_interop_buffer.blit(m_window.getWidth(), m_window.getHeight());

    renderImGui();

    // Blit rendered contents on the screen and poll events.
    m_window.endFrame();
    m_interop_buffer.unmap();

    m_window.onUpdate();
}

void Application::initCache()
{
    SimpleCudaTimer timer;

    CameraInfo cache_cam_info = m_ray_march_info.cam_info;

    if (m_rendering_path)
    {
        float next_frame_time = m_camera_path.play_time + 0.5f / m_camera_path.settings.n_frames();
        setCameraFromKeyFrame(m_camera_path.eval_camera_path(next_frame_time), cache_cam_info);
    }

    cache_cam_info.resize(m_cache_resolution);

    StepsizeInfo cache_init_stepsize_info = m_ray_march_info.stepsize_info;
    if (m_cache_double_sampling_rate)
        cache_init_stepsize_info.changeSamplingRate(2.0f, m_scene_info);

    RaymarchInfo cache_init_rm_info { cache_cam_info, {}, cache_init_stepsize_info, 0, false, true};


    timer.start();
    m_cache_renderer->initCacheBlockwise(cache_init_rm_info, m_debug_buffer);
    m_cache_renderer->swapCache();
    m_last_cache_update_pos = cache_cam_info.cam2world.getTranslation();
    float initCache_ms = timer.stopElapsed();

    // cudaPrintMemInfo();
    fmt::print("Cache Init - N Bricks: {:6} / {:6}, Samples ppx: {:6.03f}, Time: {:7.03f} ms\n",
            m_cache_renderer->_init_stats.n_bricks_set, m_cache_renderer->_init_stats.n_bricks_reserved, m_cache_renderer->_init_stats.samples_ppx, initCache_ms);

    m_init_cache = false;
    cameraMoved();
}

void Application::updateCache()
{
    while(true)
    {
        {
            std::unique_lock lock(m_update_cache_cv_mutex);
            m_update_cache_cv.wait(lock);
        }
        if (!m_running) break;

        if (m_cache_init == nullptr || m_cache_init == m_last_render_cache)
        {
            initCache();
            m_cache_init = m_cache_renderer->_cache;
        }        
    }
}

void Application::run()
{
    MY_CHECK_THROW(!m_running, "Application already running");

    m_running = true;

    auto previous = std::chrono::steady_clock::now();
    while (m_running)
    {
        if (m_rendering_path)
        {
            renderCameraPath();
            m_rendering_path = false;
        }

        m_ray_march_info.sample_index = 0;
        m_ray_march_info.cam_info = m_camera.getInfo();

        if (m_init_cache || (m_render_caching && m_auto_cache_update && (m_cache_hit_ratio < m_cache_miss_limit || 
                                                                         length(m_ray_march_info.cam_info.cam2world.getTranslation() - m_last_cache_update_pos) > m_cache_update_distance)))
        {
            if (!m_double_buffering)
            {
                initCache();
                m_camera_moved = false;
            }
            else
            {
                m_update_cache_cv.notify_one();
            }
        }
            
        void* last_render_cache = m_cache_renderer->_cache;
        while (!m_camera_moved && m_running)
        {
            auto now = std::chrono::steady_clock::now();
            m_frame_time_ms = std::chrono::duration_cast<ms>(now - previous).count();
            m_fps = static_cast<uint16_t>(1000.f / m_frame_time_ms);
            previous = now;
            
            last_render_cache = m_cache_renderer->_cache;
            renderFrame();
            m_last_render_cache = last_render_cache;
            m_cache_hit_ratio = m_cache_renderer->_render_stats.cache_samples_evaluated / m_cache_renderer->_render_stats.samples_ppx;

            m_ray_march_info.sample_index++;

            if (m_mouse_button_pressed)
                break;

            if (m_circular_camera_path)
            {
                m_camera.rotate(m_circular_speed * m_frame_time_ms, 0.f);
                break;
            }
        }

        m_camera_moved = false;
    }
    destroy();
}

void Application::renderCameraPath()
{
    m_rendering_path = true;
    const auto& rs = m_camera_path.settings;

    auto old_res = m_camera.getInfo().resolution;
    FramebufferResizeEvent e(rs.resolution.x, rs.resolution.y);
    handleFrameBufferResizeEvent(e);
    m_window.setResizeable(false);
    m_window.resize(rs.resolution.x, rs.resolution.y);

    m_ray_march_info.cam_info = m_camera.getInfo();
    float focal_x = ((float)rs.resolution.x * 0.5f) / std::tan(rs.fov_x_deg * 0.5f * (float)M_PI / 180.f);
    m_ray_march_info.cam_info.focal = float2{focal_x, focal_x};
    m_ray_march_info.next_cam_info = m_ray_march_info.cam_info;
     m_ray_march_info.motion_blur = true;
    
    SimpleCudaTimer timer;
    m_camera_path.render_frame_idx = 0;
    while(!m_abort_rendering && m_camera_path.render_frame_idx < rs.n_frames())
    {
        m_camera_path.play_time = (float)((double)m_camera_path.render_frame_idx / (double)rs.n_frames());

        setCameraFromKeyFrame(m_camera_path.eval_camera_path(m_camera_path.play_time), m_ray_march_info.cam_info);

        float next_frame_time = m_camera_path.play_time + 1.0f / rs.n_frames();
        setCameraFromKeyFrame(m_camera_path.eval_camera_path(next_frame_time), m_ray_march_info.next_cam_info);

        if (rs.render_from_cache && (!m_camera_path.settings.auto_cache_update || (m_cache_hit_ratio < m_camera_path.settings.cache_hit_limit || 
                                                                                   length(m_ray_march_info.cam_info.cam2world.getTranslation() - m_last_cache_update_pos) > m_camera_path.settings.cache_update_distance)))
        {
            timer.start();
            initCache();
            float cache_init_time_ms = timer.stopElapsed();
            std::cout << "Frame " << m_camera_path.render_frame_idx << " | Cache Init | Total: " << cache_init_time_ms << std::endl;
        }            

        m_ray_march_info.sample_index = 0;

        auto previous = std::chrono::steady_clock::now();

        timer.start();
        while (!m_abort_rendering && m_ray_march_info.sample_index < rs.spp)
        {
            auto now = std::chrono::steady_clock::now();
            m_frame_time_ms = std::chrono::duration_cast<ms>(now - previous).count();
            m_fps = static_cast<uint16_t>(1000.f / m_frame_time_ms);
            previous = now;
            renderFrame();

            m_ray_march_info.sample_index++;
        }
        float render_time_ms_total = timer.stopElapsed();
        std::cout << "Frame " << m_camera_path.render_frame_idx << " | Render | Total: " << render_time_ms_total << " Avg: " << render_time_ms_total / rs.spp << " SPP: " << rs.spp << std::endl;

        m_cache_hit_ratio = m_cache_renderer->_render_stats.cache_samples_evaluated / m_cache_renderer->_render_stats.samples_ppx;

        std::stringstream filename;
        std::string dir = std::filesystem::path(rs.filename).parent_path().string();
        filename << "render_output/frame_" << std::setfill('0') << std::setw(6) << m_camera_path.render_frame_idx << ".png";

        m_interop_buffer.writeToFile(fs::current_path() / filename.str());

        m_camera_path.render_frame_idx++;
    }

    if (!m_abort_rendering)
    {
        std::stringstream ffmpeg_command;
        ffmpeg_command << "ffmpeg -loglevel error -y -framerate " << std::to_string(rs.fps) << " -i render_output/frame_%06d.png -c:v libx264 -preset slow -crf ";
        ffmpeg_command << std::to_string(27 - rs.quality) << " -pix_fmt yuv420p \"" << rs.filename << "\"";

        auto command_str = ffmpeg_command.str();
        int ffmpeg_result = system(command_str.c_str());
    }
    else
        m_abort_rendering = false;
    
    for(auto const& entry: std::filesystem::directory_iterator{"render_output"}) {
        if (entry.is_regular_file() && entry.path().extension() == ".png")
            std::filesystem::remove(entry.path());
    }

    m_ray_march_info.motion_blur = false;
    FramebufferResizeEvent e2(old_res.x, old_res.y);
    handleFrameBufferResizeEvent(e2);
    m_window.resize(old_res.x, old_res.y);
    m_window.setResizeable(true);
}

void Application::setCameraFromKeyFrame(const CameraKeyFrame& f, CameraInfo& cam)
{
    cam.cam2world = f.m();
    cam.world2cam = cam.cam2world.inverse();
    cam.aperature = f.aperture_size;
    cam.focus_z = f.focus_distance;
}

void Application::cameraMoved()
{
    m_camera_moved = true;
}

void Application::close()
{
    printf("Stopping Application");
    m_running = false;
    m_abort_rendering = true;
}

void Application::onEvent(IEvent& event)
{
    // Check for ImGUI events;
    m_window.onEvent(event);

    if (event.isHandled())
        return;

    // Dispatch Events
    EventDispatcher dispatcher(event);

    dispatcher.dispatch<WindowCloseEvent>(BIND_EVENT_CALLBACK(Application::handleWindowClosedEvent));
    dispatcher.dispatch<FramebufferResizeEvent>(BIND_EVENT_CALLBACK(Application::handleFrameBufferResizeEvent));
    dispatcher.dispatch<MouseMovedEvent>(BIND_EVENT_CALLBACK(Application::handleMouseMovedEvent));
    dispatcher.dispatch<MouseScrollEvent>(BIND_EVENT_CALLBACK(Application::handleMouseScrolledEvent));
    dispatcher.dispatch<MouseButtonPressedEvent>(BIND_EVENT_CALLBACK(Application::handleMouseButtonPressedEvent));
    dispatcher.dispatch<MouseButtonReleasedEvent>(BIND_EVENT_CALLBACK(Application::handleMouseButtonReleasedEvent));
    dispatcher.dispatch<KeyPressedEvent>(BIND_EVENT_CALLBACK(Application::handleKeyPressedEvent));
}

bool Application::handleWindowClosedEvent(WindowCloseEvent& event)
{
    printf("Window closed\n");
    close();
    return true;
}

bool Application::handleFrameBufferResizeEvent(FramebufferResizeEvent& event)
{
    auto new_res = int2{static_cast<int>(event.getWidth()), static_cast<int>(event.getHeight())};
    std::cout << "Window resizing to: (" << new_res.x << " x " << new_res.y << ")" << std::endl;

    m_camera.resize(new_res);
    m_interop_buffer.resize(new_res);
    m_renderer->resizeRenderbuffers(new_res);

    m_init_cache = true;
    cameraMoved();
    glViewport(0, 0, event.getWidth(), event.getHeight());
    return true;
}

bool Application::handleMouseScrolledEvent(MouseScrollEvent& e)
{
    m_camera.zoom(e.getYOffset());
    cameraMoved();

    return true;
}

bool Application::handleMouseMovedEvent(MouseMovedEvent& e)
{
    if (m_window.isMouseButtonpressed(GLFW_MOUSE_BUTTON_LEFT))
    {
        float d_y = (m_last_mouse_position.y - e.getY());
        float d_x = (m_last_mouse_position.x - e.getX());

        m_camera.rotate(d_x, d_y);
        cameraMoved();
    }
    else if (m_window.isMouseButtonpressed(GLFW_MOUSE_BUTTON_MIDDLE))
    {
        float dx = m_last_mouse_position.x - e.getX();
        float dy = m_last_mouse_position.y - e.getY();

        m_camera.pan(dx * m_camera_pan_factor, dy * m_camera_pan_factor);
        cameraMoved();
    }

    m_last_mouse_position = {e.getX(), e.getY()};

    return true;
}

bool Application::handleMouseButtonReleasedEvent(MouseButtonReleasedEvent& e)
{
    m_mouse_button_pressed = false;
    return true;
}

bool Application::handleMouseButtonPressedEvent(MouseButtonPressedEvent& e)
{
    m_mouse_button_pressed = true;
    return true;
}

bool Application::handleKeyPressedEvent(KeyPressedEvent& e)
{
    if (e.getint() == GLFW_KEY_C && e.getRepeatCount() == 0)
    {
        m_render_caching = !m_render_caching;
        cameraMoved();
        std::cout << "Rendering from cache: " << (m_render_caching ? "ON" : "OFF") << std::endl;
    }
    else if (e.getint() == GLFW_KEY_V && e.getRepeatCount() == 0)
    {
        m_render_only_cache = !m_render_only_cache;
        m_cache_renderer->_render_mode = m_render_only_cache ? CachePerformanceRenderer::RenderMode::ONLY_CACHE : CachePerformanceRenderer::RenderMode::BOTH;
        cameraMoved();
        std::cout << "Visualizing ONLY cache: " << (m_render_only_cache ? "ON" : "OFF") << std::endl;
    }
    else if (e.getint() == GLFW_KEY_X && e.getRepeatCount() == 0)
    {
        m_init_cache = true;
        cameraMoved();
        std::cout << "Reinitializing cache!" << std::endl;
    }
    else if (e.getint() == GLFW_KEY_M && e.getRepeatCount() == 0)
    {
        std::cout << "Camera Matrix:" << std::endl;
        std::cout << m_camera.getInfo().cam2world.m0.x << " "  << m_camera.getInfo().cam2world.m0.y << " " << m_camera.getInfo().cam2world.m0.z << " " << m_camera.getInfo().cam2world.m0.w << std::endl;
        std::cout << m_camera.getInfo().cam2world.m1.x << " "  << m_camera.getInfo().cam2world.m1.y << " " << m_camera.getInfo().cam2world.m1.z << " " << m_camera.getInfo().cam2world.m1.w << std::endl;
        std::cout << m_camera.getInfo().cam2world.m2.x << " "  << m_camera.getInfo().cam2world.m2.y << " " << m_camera.getInfo().cam2world.m2.z << " " << m_camera.getInfo().cam2world.m2.w << std::endl;
    }

    return true;
}

void Application::renderImGui()
{
    renderImGuiCameraPath();

    if (m_rendering_path) return;
    
    ImGui::Begin("NeRF Frustum Volume Caching");
    ImGui::Text("Runtime: %.3f ms/frame (%d FPS | %d spp)", m_frame_time_ms, m_fps, m_ray_march_info.sample_index);

    ImGui::Separator();

    if(ImGui::CollapsingHeader("Cache Settings", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Text("Cache Samples / Ray: %.1f", m_cache_renderer->_render_stats.cache_samples_evaluated);
        ImGui::Text("Re-Samples / Ray: %.1f", m_cache_renderer->_render_stats.resample_samples_evaluated);
        ImGui::Text("Cache Hit Ratio: %.2f", m_cache_renderer->_render_stats.cache_samples_evaluated / m_cache_renderer->_render_stats.samples_ppx);
        ImGui::Text("Distance to Cache Viewpoint: %.2f", length(m_ray_march_info.cam_info.cam2world.getTranslation() - m_last_cache_update_pos));
        ImGui::Separator();
        if(ImGui::Checkbox("Render from Cache [C]", &m_render_caching))
            cameraMoved();

        if(!m_render_caching) ImGui::BeginDisabled();
        ImGui::SameLine();
        if(ImGui::Checkbox("Render only Cache [V]", &m_render_only_cache))
        {
            m_cache_renderer->_render_mode = m_render_only_cache ? CachePerformanceRenderer::RenderMode::ONLY_CACHE : CachePerformanceRenderer::RenderMode::BOTH;
            cameraMoved();
        }
        if(ImGui::Button("Reinitialize Cache [X]"))
        {
            m_init_cache = true;
            cameraMoved();
        }
        ImGui::Checkbox("Automatic Cache Update", &m_auto_cache_update);
        if (!m_auto_cache_update) ImGui::BeginDisabled();
        ImGui::SliderFloat("Min. Cache-Hit Ratio", &m_cache_miss_limit, 0.0f, 1.0f);
        ImGui::SliderFloat("Max. Distance", &m_cache_update_distance, 0.0f, 4.0f);
        if (!m_auto_cache_update) ImGui::EndDisabled();
        if(!m_render_caching) ImGui::EndDisabled();
    }

    ImGui::Separator();

    if (ImGui::CollapsingHeader("Camera Settings"))
    {
        if(ImGui::SliderFloat("FOV x (deg)", &(m_fov_x), 1.0f, 180.f))
        {
            float focal_x = ((float)m_camera.getInfo().resolution.x * 0.5f) / std::tan(m_fov_x * 0.5f * (float)M_PI / 180.f);
            m_camera.getInfoRef().focal = {focal_x, focal_x};
            cameraMoved();
        }
        if(ImGui::SliderFloat("Aperature", &(m_camera.getInfoRef().aperature), 0.f, 0.05f))
            cameraMoved();
        if(ImGui::SliderFloat("Focal Plane", &(m_camera.getInfoRef().focus_z), 0.1f, 5.f))
            cameraMoved();

        ImGui::Checkbox("Circular Camera Path", &m_circular_camera_path);
        
        if(!m_circular_camera_path) ImGui::BeginDisabled();  
        ImGui::SliderFloat("Circular Speed", &m_circular_speed, 0.0f, 1.f);
        if(!m_circular_camera_path) ImGui::EndDisabled();
    }
    
    ImGui::End();
}

void Application::renderImGuiCameraPath()
{
    ImGui::Begin("Camera Path");
    if (m_rendering_path)
    {
        ImGui::Text("Frame %d/%d average %.3f ms/frame (%d FPS | %d spp)", m_camera_path.render_frame_idx, m_camera_path.settings.n_frames(), m_frame_time_ms, m_fps, m_ray_march_info.sample_index);
        
        if (m_camera_path.settings.render_from_cache)
        {
            ImGui::SeparatorText("Cache Stats");

            ImGui::Text("Cache Samples / Ray: %.1f", m_cache_renderer->_render_stats.cache_samples_evaluated);
            ImGui::Text("Re-Samples / Ray: %.1f", m_cache_renderer->_render_stats.resample_samples_evaluated);
            ImGui::Text("Cache Hit Ratio: %.2f", m_cache_renderer->_render_stats.cache_samples_evaluated / m_cache_renderer->_render_stats.samples_ppx);
            ImGui::Text("Distance to last Cache Generation Point: %.3f", length(m_ray_march_info.cam_info.cam2world.getTranslation() - m_last_cache_update_pos));
        }

        ImGui::Separator();

        if (ImGui::Button("Abort"))
            m_abort_rendering = true;

        ImGui::End();
        return;
    }

    if (ImGui::Button("Load Camera Path")) {
        IGFD::FileDialogConfig config;config.path = ".";
        ImGuiFileDialog::Instance()->OpenDialog("ChooseCameraPathFile", "Choose File", ".json", config);
    }
    if (ImGuiFileDialog::Instance()->Display("ChooseCameraPathFile")) {
        if (ImGuiFileDialog::Instance()->IsOk()) {
            std::string filePath = ImGuiFileDialog::Instance()->GetFilePathName();

            m_camera_path.load(filePath);
        }
        // close
        ImGuiFileDialog::Instance()->Close();
    }
    ImGui::SameLine();
    if (ImGui::Button("Add KeyFrame from Camera"))
    {
        m_camera_path.key_frames.emplace_back(m_camera.getInfo().cam2world, m_camera.getInfo().aperature, m_camera.getInfo().focus_z);
    }
    if (!m_camera_path.key_frames.empty())
    {    
        if (ImGui::Button("Save Camera Path")) {
            IGFD::FileDialogConfig config;config.path = ".";
            ImGuiFileDialog::Instance()->OpenDialog("SaveCameraPathFile", "Choose File", ".json", config);
        }
        if (ImGuiFileDialog::Instance()->Display("SaveCameraPathFile")) {
            if (ImGuiFileDialog::Instance()->IsOk()) {
                std::string filePath = ImGuiFileDialog::Instance()->GetFilePathName();

                m_camera_path.save(filePath);
            }
            // close
            ImGuiFileDialog::Instance()->Close();
        }

        ImGui::SameLine();
        
        if (ImGui::Button("Reset Camera Path"))
            ImGui::OpenPopup("Delete Current Path");
        
        if (ImGui::BeginPopupModal("Delete Current Path", nullptr))
        {
            if (ImGui::Button("Yes")) {
                m_camera_path.key_frames.clear();
                ImGui::CloseCurrentPopup();
            }
            if (ImGui::Button("Cancel"))
                ImGui::CloseCurrentPopup();
            ImGui::EndPopup();
        }

        if (ImGui::CollapsingHeader("Render Settings"))
        {
            ImGui::InputFloat("Duration (s)", &(m_camera_path.settings.duration_seconds));
            ImGui::InputFloat("FPS (frames/second)", &(m_camera_path.settings.fps));
            ImGui::InputInt("SPP (samples/pixel)", &(m_camera_path.settings.spp));
            ImGui::SliderInt("Video Output Quality", &(m_camera_path.settings.quality), 0, 10);

            ImGui::SeparatorText("Camera Settings");

            if(ImGui::InputInt2("Resolution", &(m_camera_path.settings.resolution.x)))
            {
                m_camera_path.settings.resolution.x = round(m_camera_path.settings.resolution.x / 2) * 2;
                m_camera_path.settings.resolution.y = round(m_camera_path.settings.resolution.y / 2) * 2;
            }
            ImGui::SliderFloat("FOV x (deg)", &(m_camera_path.settings.fov_x_deg), 1.0f, 180.f);

            ImGui::SeparatorText("Cache Settings");

            ImGui::Checkbox("Render from Cache", &(m_camera_path.settings.render_from_cache));
            if (!m_camera_path.settings.render_from_cache) ImGui::BeginDisabled();
            ImGui::SameLine();
            ImGui::Checkbox("Automatic Cache Update", &(m_camera_path.settings.auto_cache_update));
            if (!m_camera_path.settings.auto_cache_update) ImGui::BeginDisabled();
            ImGui::SliderFloat("Min. Cache Hit Ratio", &(m_camera_path.settings.cache_hit_limit), 0.0f, 1.0f);
            ImGui::SliderFloat("Max. Distance before Cache Update", &(m_camera_path.settings.cache_update_distance), 0.0f, 4.0f);
            if (!m_camera_path.settings.auto_cache_update) ImGui::EndDisabled();
            if (!m_camera_path.settings.render_from_cache) ImGui::EndDisabled();

            ImGui::Separator();
        }
        // ImGui::SliderFloat("Shutter fraction", &(m_camera_path.settings.shutter_fraction), 0.0f, 1.0f);
        
        if (ImGui::Button("Render Video")) {
            IGFD::FileDialogConfig config;config.path = ".";
            ImGuiFileDialog::Instance()->OpenDialog("RenderPathFile", "Choose File", ".mp4", config);
        }
        if (ImGuiFileDialog::Instance()->Display("RenderPathFile")) {
            if (ImGuiFileDialog::Instance()->IsOk()) {
                m_camera_path.settings.filename = ImGuiFileDialog::Instance()->GetFilePathName();
                m_rendering_path = true;
            }
            // close
            ImGuiFileDialog::Instance()->Close();
        }
    }
    ImGui::End();
}