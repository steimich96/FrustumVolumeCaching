#pragma once

#include <owl/owl.h>
#include <cuda_runtime.h>

#include "common.h"
#include "renderer_common.h"


struct InitRayPayloadsOptixRayGenParams
{
    OptixTraversableHandle grid_geom;

    const RaymarchInfo rm_info;
    const SceneInfo scene_info;

    int* alive_buffer;
    BaseRayInitInfo *init_infos;
    SegmentedRayPayload *payloads;
    BaseRayResult *results;
    int *ray_order_buffer;
};

struct InitRayPayloadsProgram
{
    InitRayPayloadsProgram() {};

    void __host__ init(OWLContext context, OWLModule module, OWLGeomType triangle_geom_type)
    {
        OWLVarDecl miss_prog_vars[] = {{nullptr /* sentinel */}};
        _miss_program = owlMissProgCreate(context, module, "TriangleMeshMP", sizeof(MissParams), miss_prog_vars, -1);

        OWLVarDecl raygen_vars[] = 
        {
            {"grid_geom", OWL_GROUP, OWL_OFFSETOF(InitRayPayloadsOptixRayGenParams, grid_geom)},

            {"rm_info", OWL_USER_TYPE(RaymarchInfo), OWL_OFFSETOF(InitRayPayloadsOptixRayGenParams, rm_info)},
            {"scene_info", OWL_USER_TYPE(SceneInfo), OWL_OFFSETOF(InitRayPayloadsOptixRayGenParams, scene_info)},

            {"alive_buffer", OWL_RAW_POINTER, OWL_OFFSETOF(InitRayPayloadsOptixRayGenParams, alive_buffer)},
            {"init_infos", OWL_RAW_POINTER, OWL_OFFSETOF(InitRayPayloadsOptixRayGenParams, init_infos)},
            {"payloads", OWL_RAW_POINTER, OWL_OFFSETOF(InitRayPayloadsOptixRayGenParams, payloads)},
            {"results", OWL_RAW_POINTER, OWL_OFFSETOF(InitRayPayloadsOptixRayGenParams, results)},
            {"ray_order_buffer", OWL_RAW_POINTER, OWL_OFFSETOF(InitRayPayloadsOptixRayGenParams, ray_order_buffer)},
            {nullptr /* sentinel */}
        };
        _raygen_program = owlRayGenCreate(context, module, "InitRayPayloadsRGP", sizeof(InitRayPayloadsOptixRayGenParams), raygen_vars, -1);

        owlGeomTypeSetClosestHit(triangle_geom_type, 0, module, "TriangleMeshCHP");

        _initialized = true;
    }

    void __host__ dummyLaunch(OWLContext context)
    {
        // Launches the raygen kernel once because of some startup costs on very first launch
        RaymarchInfo dummy_raymarch_info = { cam_info: {resolution: make_int2(0)}}; // so kernel immediately exits
        owlRayGenSetRaw(_raygen_program, "rm_info", &dummy_raymarch_info);
        owlBuildSBT(context);

        owlRayGenLaunch2D(_raygen_program, 1, 1);
    }

    void __host__ launch(OWLContext context, OWLGroup grid_geom, RaymarchInfo &rm_info, SceneInfo &scene_info,
                         int* alive_buffer, BaseRayInitInfo* init_infos, SegmentedRayPayload* payloads, BaseRayResult* results,
                         int* ray_order_buffer = nullptr)
    {
        owlRayGenSetGroup(_raygen_program, "grid_geom", grid_geom);

        owlRayGenSetRaw(_raygen_program, "rm_info", &rm_info);
        owlRayGenSetRaw(_raygen_program, "scene_info", &scene_info);

        owlRayGenSetPointer(_raygen_program, "alive_buffer", (const void *)alive_buffer);
        owlRayGenSetPointer(_raygen_program, "init_infos", (const void *)init_infos);
        owlRayGenSetPointer(_raygen_program, "payloads", (const void *)payloads);
        owlRayGenSetPointer(_raygen_program, "results", (const void *)results);
        owlRayGenSetPointer(_raygen_program, "ray_order_buffer", (const void *)ray_order_buffer);
        owlBuildSBT(context);

        owlRayGenLaunch2D(_raygen_program, rm_info.cam_info.resolution.x, rm_info.cam_info.resolution.y);
    }  

    bool _initialized = false;

    OWLMissProg _miss_program;
    OWLRayGen _raygen_program;
};