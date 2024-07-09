#include "programs/sample_segments.h"

#include <optix_device.h>

#include "raymarch_common.h"
#include "util/owl_device_helper.h"

OPTIX_RAYGEN_PROGRAM(SampleSegmentsRGP)()
{
    const SampleSegmentsOptixRayGenParams &params = owl::getProgramData<SampleSegmentsOptixRayGenParams>();
    const int curr_subiter_idx = owl::getLaunchIndex().x;

    if (curr_subiter_idx >= params.n_alive)
        return;

    if (!params.alive_buffer[curr_subiter_idx])
        return;

    const BaseRayInitInfo ray_init = params.ray_inits[curr_subiter_idx];
    SegmentedRayPayload payload = params.payloads[curr_subiter_idx];

    float t1 = payload.curr_t1;
    owl::Ray owl_ray(ray_init.origin, ray_init.dir, t1, ray_init.far + 1e-5f);

    float t0 = t1;
    float dt = calculate_stepsize(t0, params.rm_info.stepsize_info);
    float t_mid = t0 + dt * 0.5f;


    Segment segment;
    if (t_mid < payload.curr_segment_end)
    {
        segment = Segment { payload.curr_t1, payload.curr_segment_end, true };
    }
    else
    {
        // segment = computeIntersectionSegment(owl_ray, params.grid_geom);
        // segment.begin = valid_t_from_t(segment.begin, ray_init.near_i, params.rm_info.stepsize_info);

        Segment tmp_segment = computeIntersectionSegment(owl_ray, params.grid_geom);
        while (tmp_segment.hit)
        {
            if (steps_in_segment(tmp_segment, params.rm_info.stepsize_info) > 0)
            {
                printf("RAYINIT needs near_i");
                segment.begin = 0.0f; //valid_t_from_t(tmp_segment.begin, ray_init.near_i, params.rm_info.stepsize_info);
                segment.end = tmp_segment.end;
                segment.hit = true;
                break;
            }

            owl_ray.tmin = tmp_segment.end + 1e-8f;
            tmp_segment = computeIntersectionSegment(owl_ray, params.grid_geom);
        }
    }

    params.alive_buffer[curr_subiter_idx] = segment.hit;

    payload.curr_t1 = segment.begin;
    payload.curr_segment_end = segment.end;

    params.payloads[curr_subiter_idx] = payload;
}