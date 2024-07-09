
#pragma once

#include "common.h"
#include "util/mesh.h"

#include <filesystem>

int createMeshFromGridCompact(TriangleMesh& mesh, SceneInfo scene_info, CudaBuffer<uint8_t>& grid);
void writeMeshToObj(TriangleMesh& mesh, std::filesystem::path data_dir, const char filename[]);

#ifdef RTX_ENABLED
int createMeshFromGridCompact(TriangleMeshOwl& mesh, SceneInfo scene_info, CudaBuffer<uint8_t>& grid, OWLContext context);
void writeMeshToObj(TriangleMeshOwl& mesh, std::filesystem::path data_dir, const char filename[]);
#endif
