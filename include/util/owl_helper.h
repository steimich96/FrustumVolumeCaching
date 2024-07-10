/*
 * Copyright (C) 2024, Michael Steiner, Graz University of Technology.
 * This code is licensed under the MIT license.
 */

#pragma once

#include "mesh.h"

// OTHER
struct OptixTrianglesGeomData
{
    int3 *index;
    float3 *vertex;
};

inline OWLGeomType createGeomType(OWLContext& context)
{
    OWLVarDecl trianglesGeomVars[] = {
        {"index", OWL_BUFPTR, OWL_OFFSETOF(OptixTrianglesGeomData, index)},
        {"vertex", OWL_BUFPTR, OWL_OFFSETOF(OptixTrianglesGeomData, vertex)}};

    return owlGeomTypeCreate(context, OWL_TRIANGLES, sizeof(OptixTrianglesGeomData), trianglesGeomVars, 2);
}

inline OWLGroup buildAccel(TriangleMeshOwl& mesh, OWLGeomType &geom_type, OWLContext& context)
{
    OWLGeom trianglesGeom = owlGeomCreate(context, geom_type);

    owlGeomSetBuffer(trianglesGeom, "index", mesh.indices);
    owlGeomSetBuffer(trianglesGeom, "vertex", mesh.vertices);

    owlTrianglesSetVertices(trianglesGeom, mesh.vertices, mesh.total_n_vertices, sizeof(float3), 0);
    owlTrianglesSetIndices(trianglesGeom, mesh.indices, mesh.n_triangles, sizeof(int3), 0);

    OWLGroup trianglesGroup = owlTrianglesGeomGroupCreate(context, 1, &trianglesGeom);
    owlGroupBuildAccel(trianglesGroup);
    OWLGroup accel = owlInstanceGroupCreate(context, 1, &trianglesGroup);
    owlGroupBuildAccel(accel);

    return accel;
}
