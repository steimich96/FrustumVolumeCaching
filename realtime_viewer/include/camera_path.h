#pragma once

#include "common.h"

#include <tiny-cuda-nn/common.h>
#include <json/json.hpp>

#include <vector>
#include <chrono>
#include <string>

// Based on: https://github.com/NVlabs/instant-ngp/blob/master/include/neural-graphics-primitives/camera_path.h
struct CameraKeyFrame {
    tcnn::quat R;
    tcnn::vec3 T;

    float aperture_size;
    float focus_distance;

    CameraKeyFrame() = default;
    CameraKeyFrame(tcnn::quat r, tcnn::vec3 t, float ap, float fd) : R(r), T(t), aperture_size(ap), focus_distance(fd) {};
    CameraKeyFrame(CameraMatrix cam, float ap, float fd) : aperture_size(ap), focus_distance(fd) {
        T = {cam.m0.w, cam.m1.w, cam.m2.w};
        R = tcnn::quat(tcnn::mat3{cam.m0.x, cam.m1.x, cam.m2.x, cam.m0.y, cam.m1.y, cam.m2.y, cam.m0.z, cam.m1.z, cam.m2.z});
    }

    CameraMatrix m() const {
        auto rot = to_mat3(normalize(R));
        return CameraMatrix{{rot[0].x, rot[1].x, rot[2].x, T.x}, {rot[0].y, rot[1].y, rot[2].y, T.y}, {rot[0].z, rot[1].z, rot[2].z, T.z}};
    }

    CameraKeyFrame operator*(float f) const { return {R*f, T*f, aperture_size*f, focus_distance*f}; }
    CameraKeyFrame operator+(const CameraKeyFrame& rhs) const {
		tcnn::quat Rr = rhs.R;
		if (dot(Rr, R) < 0.0f) Rr = -Rr;
		return {R+Rr, T+rhs.T, aperture_size+rhs.aperture_size, focus_distance+rhs.focus_distance};
	}
};

CameraKeyFrame normalize(const CameraKeyFrame& p0);
CameraKeyFrame spline(float t, const CameraKeyFrame& p0, const CameraKeyFrame& p1, const CameraKeyFrame& p2, const CameraKeyFrame& p3);

// Based on: https://github.com/NVlabs/instant-ngp/blob/master/include/neural-graphics-primitives/camera_path.h
struct CameraPath {
    std::vector<CameraKeyFrame> key_frames;

    struct RenderSettings {
        bool render_from_cache = false;
		bool auto_cache_update = false;
		float cache_update_distance = 1.1f;
		float cache_hit_limit = 0.5f;
		int2 resolution = {1920, 1080};
		float fov_x_deg = 60.f;
		int spp = 8;
		float fps = 60.0f;
		float duration_seconds = 5.0f;
		float shutter_fraction = 0.5f;
		int quality = 10;

		uint32_t n_frames() const {
			return (uint32_t)((double)duration_seconds * fps);
		}

		float frame_seconds() const {
			return 1.0f / (duration_seconds * fps);
		}

		float frame_milliseconds() const {
			return 1000.0f / (duration_seconds * fps);
		}

		std::string filename = "video.mp4";
	};

    RenderSettings settings;

    float play_time = 0.f;
    std::chrono::time_point<std::chrono::steady_clock> render_start_time;

    bool rendering = false;
	uint32_t render_frame_idx = 0;

    CameraInfo render_frame_end_camera;

    const CameraKeyFrame& get_keyframe(int i) {
        return key_frames[clamp(i, 0, (int)key_frames.size()-1)];
	}

	CameraKeyFrame eval_camera_path(float t) {
		if (key_frames.empty())
			return {};

		t *= (float)key_frames.size()-1;
		int t1 = (int)floorf(t);
		return spline(t-floorf(t), get_keyframe(t1-1), get_keyframe(t1), get_keyframe(t1+1), get_keyframe(t1+2));
	}

    bool load(const std::string& path);
    void save(const std::string& path);
};