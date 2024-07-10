/*
 * Copyright (C) 2024, Michael Steiner, Graz University of Technology.
 * This code is licensed under the MIT license.
 */

#pragma once

#include <owl/owl.h>
#include <cuda_runtime.h>

#include "common.h"
#include "renderer_common.h"


struct SampleSegmentsOptixRayGenParams
{
    OptixTraversableHandle grid_geom;

    int n_alive;
    int n_steps_curr_subiter;

    RaymarchInfo rm_info;
    SceneInfo scene_info;

    const BaseRayInitInfo* ray_inits;
    SegmentedRayPayload* payloads;
    int* alive_buffer;
};

struct SampleSegmentsProgram
{
    SampleSegmentsProgram() {};

    void __host__ init(OWLContext context, OWLModule module, OWLGeomType triangle_geom_type)
    {
        OWLVarDecl miss_prog_vars[] = {{nullptr /* sentinel */}};
        _miss_program = owlMissProgCreate(context, module, "TriangleMeshMP", sizeof(MissParams), miss_prog_vars, -1);

        OWLVarDecl raygen_vars[] = 
        {
            {"grid_geom", OWL_GROUP, OWL_OFFSETOF(SampleSegmentsOptixRayGenParams, grid_geom)},

            {"n_alive", OWL_INT, OWL_OFFSETOF(SampleSegmentsOptixRayGenParams, n_alive)},
            {"n_steps_curr_subiter", OWL_INT, OWL_OFFSETOF(SampleSegmentsOptixRayGenParams, n_steps_curr_subiter)},

            {"rm_info", OWL_USER_TYPE(RaymarchInfo), OWL_OFFSETOF(SampleSegmentsOptixRayGenParams, rm_info)},
            {"scene_info", OWL_USER_TYPE(SceneInfo), OWL_OFFSETOF(SampleSegmentsOptixRayGenParams, scene_info)},

            {"ray_inits", OWL_RAW_POINTER, OWL_OFFSETOF(SampleSegmentsOptixRayGenParams, ray_inits)},
            {"payloads", OWL_RAW_POINTER, OWL_OFFSETOF(SampleSegmentsOptixRayGenParams, payloads)},
            {"alive_buffer", OWL_RAW_POINTER, OWL_OFFSETOF(SampleSegmentsOptixRayGenParams, alive_buffer)},
            {nullptr /* sentinel */}
        };
        _raygen_program = owlRayGenCreate(context, module, "SampleSegmentsRGP", sizeof(SampleSegmentsOptixRayGenParams), raygen_vars, -1);

        owlGeomTypeSetClosestHit(triangle_geom_type, 0, module, "TriangleMeshCHP");

        _initialized = true;
    }

    void __host__ dummyLaunch(OWLContext context)
    {
        // Launches the raygen kernel once because of some startup costs on very first launch
        int zero = 0; // so kernel immediately exits
        owlRayGenSet1i(_raygen_program, "n_alive", (const int &) zero);
        owlBuildSBT(context);

        owlRayGenLaunch2D(_raygen_program, 1, 1);
    }

    void __host__ launch(OWLContext context, OWLGroup grid_geom, int n_alive, int n_steps_curr_subiter, RaymarchInfo &rm_info, SceneInfo &scene_info,
                         BaseRayInitInfo* init_infos, SegmentedRayPayload* payloads, int* alive_buffer)
    {
        owlRayGenSetGroup(_raygen_program, "grid_geom", grid_geom);

        owlRayGenSet1i(_raygen_program, "n_alive", (const int &) n_alive);
        owlRayGenSet1i(_raygen_program, "n_steps_curr_subiter", (const int &) n_steps_curr_subiter);

        owlRayGenSetRaw(_raygen_program, "rm_info", &rm_info);
        owlRayGenSetRaw(_raygen_program, "scene_info", &scene_info);

        owlRayGenSetPointer(_raygen_program, "ray_inits", (const void *)init_infos);
        owlRayGenSetPointer(_raygen_program, "payloads", (const void *)payloads);
        owlRayGenSetPointer(_raygen_program, "alive_buffer", (const void *)alive_buffer);
        owlBuildSBT(context);

        owlRayGenLaunch2D(_raygen_program, n_alive, 1);
    } 
    

    bool _initialized = false;

    OWLMissProg _miss_program;
    OWLRayGen _raygen_program;
};