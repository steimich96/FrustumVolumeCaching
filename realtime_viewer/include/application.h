#pragma once

#include "event.h"
#include "events/window_events.h"
#include "window.h"
#include "trackball_camera.h"
#include "camera_path.h"

#include <common.h>
#include <cache.h>
#include <loader.h>
#include <performance_renderer.h>
#include <cache_performance_renderer.h>

#include <memory>
#include <mutex> 
#include <thread>
#include <condition_variable>
#include <atomic>

class Application 
{
public:
    Application(const Application&) = delete;

    Application(int2 resolution, std::filesystem::path data_dir, CacheSettings cache_settings, CameraInfo camera_info, CameraMatrix initial_pose, float fov_x, CameraPath cam_path = CameraPath(), bool double_buffering = true);

    ~Application();

    void run();
    void renderCameraPath();

protected:
    Application();

private:

    void close();
    void destroy();

    void initCache();
    void updateCache();

    void renderFrame();

    void onEvent(IEvent& event);

    void cameraMoved();

    void renderImGui();
    void renderImGuiCameraPath();

    void setCameraFromKeyFrame(const CameraKeyFrame& f, CameraInfo& cam);

    bool handleWindowClosedEvent(WindowCloseEvent& event);
    bool handleFrameBufferResizeEvent(FramebufferResizeEvent& event);
    bool handleMouseScrolledEvent(MouseScrollEvent& e);
    bool handleMouseMovedEvent(MouseMovedEvent& e);
    bool handleMouseButtonReleasedEvent(MouseButtonReleasedEvent& e);
    bool handleMouseButtonPressedEvent(MouseButtonPressedEvent& e);
    bool handleKeyPressedEvent(KeyPressedEvent& e);

    std::atomic<bool> m_running;

    float2 m_last_mouse_position;

    float m_camera_scroll_factor = 0.05f;
    float m_camera_move_factor = 0.003f;
    float m_camera_pan_factor = 0.005f;
    bool m_camera_moved = false;
    bool m_mouse_button_pressed = false;

    int2 m_cache_resolution;
    float m_cache_hit_ratio;
    float m_cache_miss_limit = 0.5f;
    float m_cache_update_distance = 1.1f;
    float3 m_last_cache_update_pos;
    bool m_render_caching = false;
    bool m_render_only_cache = false;
    bool m_init_cache = false;
    bool m_double_buffering = false;
    bool m_auto_cache_update = false;
    bool m_cache_double_sampling_rate = true;
    void* m_last_render_cache = nullptr;
    void* m_cache_init = nullptr;

    bool m_circular_camera_path = false;
    float m_circular_speed = 0.05f;

    bool m_rendering_path = false;
    bool m_abort_rendering = false;

    uint16_t m_fps;
    float m_frame_time_ms;

    std::mutex m_update_cache_cv_mutex;
    std::condition_variable m_update_cache_cv;
    std::thread m_cache_update_thread;

    float m_fov_x;

    Window m_window;
    TrackballCamera m_camera;
    InteropBuffer m_interop_buffer;
    InteropBuffer m_debug_buffer;
    std::shared_ptr<CachePerformanceRenderer> m_cache_renderer;
    std::shared_ptr<PerformanceRenderer> m_renderer;
    Cache m_cache;
    Cache m_cache_next;
    LoaderData m_data_loader;
    SceneInfo m_scene_info;
    RaymarchInfo m_ray_march_info;
    CameraPath m_camera_path;
};