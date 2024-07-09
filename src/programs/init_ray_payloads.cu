#include "programs/init_ray_payloads.h"

#include <optix_device.h>

#include "raymarch_common.h"
#include "util/random_val.h"
#include "util/owl_device_helper.h"

OPTIX_RAYGEN_PROGRAM(InitRayPayloadsRGP)()
{
    const InitRayPayloadsOptixRayGenParams &params = owl::getProgramData<InitRayPayloadsOptixRayGenParams>();
    const int2 pixel_idx = owl::getLaunchIndex();

    const int2 resolution = params.rm_info.cam_info.resolution;
    if (pixel_idx.x >= resolution.x || pixel_idx.y >= resolution.y)
        return;

    const float near = params.rm_info.stepsize_info.near;
    const float far = params.rm_info.stepsize_info.far;

    const int ray_id = pixel_idx.y * resolution.x + pixel_idx.x;
    const int ray_order = (params.ray_order_buffer != nullptr) ? params.ray_order_buffer[ray_id] : ray_id;

    float2 pixel_offset = {0.5f, 0.5f};

    if (!params.rm_info.deterministic)
        pixel_offset = ld_random_pixel_offset(params.rm_info.sample_index, ray_id);

    const float2 screen_point = make_float2(pixel_idx.x + pixel_offset.x, pixel_idx.y + pixel_offset.y);
    const FrustumRay ray = generateRay(ray_id, screen_point, params.rm_info);

    float aabb_t0, aabb_t1;
    bool hits_aabb = ray_aabb_intersect(ray, params.scene_info.aabb_from, params.scene_info.aabb_to, aabb_t0, aabb_t1);

    Segment segment;
    float first_segment_end;
    segment.hit = false;

    if (hits_aabb)
    {
        owl::Ray owl_ray(ray.origin, ray.dir, max(near, aabb_t0 - 1e-5f), aabb_t1 + 1e-5f);

        Segment tmp_segment = computeIntersectionSegment(owl_ray, params.grid_geom);
        while (tmp_segment.hit)
        {
            if (steps_in_segment(tmp_segment, params.rm_info.stepsize_info) > 0)
            {
                if (segment.hit)
                {
                    segment.end = tmp_segment.end;
                }
                else
                {
                    segment = tmp_segment;
                    segment.end = aabb_t1;
                    first_segment_end = tmp_segment.end;
                    break;
                }
            }

            owl_ray.tmin = tmp_segment.end + 1e-8f;
            tmp_segment = computeIntersectionSegment(owl_ray, params.grid_geom);
        }
    }

    params.alive_buffer[ray_order] = segment.hit;

    float i_offset = params.rm_info.deterministic ? 0.0f : ld_random_t_offset(params.rm_info.sample_index, ray_id * 56924617);
    float jittered_near_i = params.rm_info.stepsize_info.near_i + i_offset;
    params.init_infos[ray_order] = {
        origin : ray.origin,
        dir : ray.dir,
        ray_id : ray_id,
        far : segment.hit ? min(segment.end, far) : far,
        //near_i : jittered_near_i needed if using SAMPLE_WITH_RTX
    };

    float start_t = segment.hit ? valid_t_from_t(segment.begin, jittered_near_i, params.rm_info.stepsize_info) : near;
    params.payloads[ray_order] = {
        start_t,
        segment.hit ? first_segment_end : 0.0f
    };

    params.results[ray_order] = {
        {0.0f, 0.0f, 0.0f},
        1.0f
    };
}