/*
 * Copyright (C) 2024, Michael Steiner, Graz University of Technology.
 * This code is licensed under the MIT license.
 */

#pragma once

#include "buffer.h"

#ifdef RTX_ENABLED
#include "owl/owl_host.h"

struct TriangleMeshOwl
{
    OWLBuffer vertices;
    OWLBuffer indices;

    int total_n_vertices;
    int n_triangles;
};
#endif

struct TriangleMesh
{
    CudaBuffer<float3> vertices;
    CudaBuffer<int3> indices;

    int total_n_vertices;
    int n_triangles;
};