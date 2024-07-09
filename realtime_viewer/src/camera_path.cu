#include "camera_path.h"

#include <tiny-cuda-nn/vec_json.h>

#include <iostream>
#include <fstream>

// Based on: https://github.com/NVlabs/instant-ngp/blob/master/src/camera_path.cu
CameraKeyFrame lerp(const CameraKeyFrame& p0, const CameraKeyFrame& p1, float t, float t0, float t1) {
	t = (t - t0) / (t1 - t0);
	tcnn::quat R1 = p1.R;

	// take the short path
	if (dot(R1, p0.R) < 0.0f)  {
		R1 = -R1;
	}

	return {
		normalize(slerp(p0.R, R1, t)),
		p0.T + (p1.T - p0.T) * t,
		p0.aperture_size + (p1.aperture_size - p0.aperture_size) * t,
		p0.focus_distance + (p1.focus_distance - p0.focus_distance) * t,
	};
}

// Based on: https://github.com/NVlabs/instant-ngp/blob/master/src/camera_path.cu
CameraKeyFrame normalize(const CameraKeyFrame& p0)
{
	CameraKeyFrame result = p0;
	result.R = normalize(result.R);
	return result;
}

// Based on: https://github.com/NVlabs/instant-ngp/blob/master/src/camera_path.cu
CameraKeyFrame spline(float t, const CameraKeyFrame& p0, const CameraKeyFrame& p1, const CameraKeyFrame& p2, const CameraKeyFrame& p3)
{
	// Catmull-Rom
	CameraKeyFrame q0 = lerp(p0, p1, t, -1.f, 0.f);
	CameraKeyFrame q1 = lerp(p1, p2, t,  0.f, 1.f);
	CameraKeyFrame q2 = lerp(p2, p3, t,  1.f, 2.f);
	CameraKeyFrame r0 = lerp(q0, q1, t, -1.f, 1.f);
	CameraKeyFrame r1 = lerp(q1, q2, t,  0.f, 2.f);
	return lerp(r0, r1, t, 0.f, 1.f);
	// cubic bspline
	// float tt = t*t;
	// float ttt = t*t*t;
	// float a = (1-t)*(1-t)*(1-t)*(1.f/6.f);
	// float b = (3.f*ttt-6.f*tt+4.f)*(1.f/6.f);
	// float c = (-3.f*ttt+3.f*tt+3.f*t+1.f)*(1.f/6.f);
	// float d = ttt*(1.f/6.f);
	// return normalize(p0 * a + p1 * b + p2 * c + p3 * d);
};

void to_json(nlohmann::json& j, const CameraPath::RenderSettings& s) 
{
	j = nlohmann::json{
		{"resolution", {s.resolution.x, s.resolution.y}},
		{"fov_x", s.fov_x_deg},
		{"render_from_cache", s.render_from_cache},
		{"auto_cache_update", s.auto_cache_update},
		{"cache_update_distance", s.cache_update_distance},
		{"cache_min_hit_ratio", s.cache_hit_limit},
		{"spp", s.spp},
		{"fps", s.fps},
		{"duration", s.duration_seconds},
		{"shutter_fraction", s.shutter_fraction},
		{"quality", s.quality}
	};
}

void from_json(const nlohmann::json& j, CameraPath::RenderSettings& s)
{
	std::vector<int> res = j["resolution"].get<std::vector<int>>();
	s.resolution = make_int2(res[0], res[1]);
    j.at("render_from_cache").get_to(s.render_from_cache);
	j.at("auto_cache_update").get_to(s.auto_cache_update);
	j.at("cache_update_distance").get_to(s.cache_update_distance);
	j.at("cache_min_hit_ratio").get_to(s.cache_hit_limit);
	j.at("fov_x").get_to(s.fov_x_deg);
    j.at("spp").get_to(s.spp);
	j.at("fps").get_to(s.fps);
	j.at("duration").get_to(s.duration_seconds);
	j.at("shutter_fraction").get_to(s.shutter_fraction);
	j.at("quality").get_to(s.quality);
}

void to_json(nlohmann::json& j, const CameraKeyFrame& p) 
{
	j = nlohmann::json{
		{"R", p.R},
		{"T", p.T},
		{"aperture_size", p.aperture_size},
		{"focus_distance", p.focus_distance}
	};
}

void from_json(const nlohmann::json& j, CameraKeyFrame& p)
{
    p.R = j.at("R");
    p.T = j.at("T");

    j.at("aperture_size").get_to(p.aperture_size);
    j.at("focus_distance").get_to(p.focus_distance);
}

bool CameraPath::load(const std::string& path)
{
    std::ifstream f(path);
    if (!f) {
		std::cout << "Camera Path File []" << path << "] does not exist!" << std::endl;
		return false;
	}

    nlohmann::json j;
    f >> j;

    key_frames.clear();

	if (j.contains("settings")) settings = j["settings"];
    if (j.contains("path")) for (auto& el : j["path"]) {
        CameraKeyFrame p;
        from_json(el, p);
		key_frames.push_back(p);
	}
	return true;
}

void CameraPath::save(const std::string& path)
{
    nlohmann::json j = {
		{"settings", settings},
		{"path", key_frames},
	};
    std::ofstream f(path);
    f << j;
}