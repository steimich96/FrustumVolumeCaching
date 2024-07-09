
#pragma once

#include "common.h"
#include <owl/owl_device.h>

struct Payload
{
    float t;
    bool hit;
};

inline __device__ Segment computeIntersectionSegment(owl::Ray ray, OptixTraversableHandle geom)
{
    Segment segment {1e10f, 1e10f};

    Payload payload;
    payload.hit = false;

    owl::traceRay(geom, ray, payload, OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_CULL_FRONT_FACING_TRIANGLES);
    segment.hit = payload.hit;

    if (!segment.hit)
        return segment;

    segment.end = payload.t;
    const float frustum_segment_t1 = payload.t;

    payload.hit = false;
    owl::traceRay(geom, ray, payload, OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES);

    segment.begin = payload.hit && (payload.t <= frustum_segment_t1) ? payload.t : ray.tmin;
    return segment;
}

OPTIX_CLOSEST_HIT_PROGRAM(TriangleMeshCHP)()
{
    Payload &payload = owl::getPRD<Payload>();

    const float t = optixGetRayTmax();
    payload.t = t;
    payload.hit = true;

    return;
}

OPTIX_MISS_PROGRAM(TriangleMeshMP)()
{    
    return;
}