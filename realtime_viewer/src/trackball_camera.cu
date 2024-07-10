/*
 * Copyright (C) 2024, Michael Steiner, Graz University of Technology.
 * This code is licensed under the MIT license.
 */

#include "trackball_camera.h"

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/polar_coordinates.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <cmath>

TrackballCamera::TrackballCamera(glm::vec3 target, float radius, CameraInfo cam)
    : m_target(target)
    , m_radius(radius)
    , m_cam(cam)
    , m_up(0.f, 1.0f, 0.f)
    , m_phi(0.1f)
    , m_theta(0.1f)
{

}

glm::mat3x3 TrackballCamera::getRotation()
{
    float values[9] = {
        m_cam.cam2world.m0.x, m_cam.cam2world.m1.x, m_cam.cam2world.m2.x,
        m_cam.cam2world.m0.y, m_cam.cam2world.m1.y, m_cam.cam2world.m2.y,
        m_cam.cam2world.m0.z, m_cam.cam2world.m1.z, m_cam.cam2world.m2.z,
    };

    return glm::make_mat3(values);
}

void TrackballCamera::addRotation(const glm::mat3x3& rot_mat)
{
    float values[12] = {
        m_cam.cam2world.m0.x, m_cam.cam2world.m1.x, m_cam.cam2world.m2.x, 
        m_cam.cam2world.m0.y, m_cam.cam2world.m1.y, m_cam.cam2world.m2.y,
        m_cam.cam2world.m0.z, m_cam.cam2world.m1.z, m_cam.cam2world.m2.z,
        m_cam.cam2world.m0.w, m_cam.cam2world.m1.w, m_cam.cam2world.m2.w,
    };

    glm::mat4x3 cam_mat = glm::make_mat4x3(values);

    auto new_mat = rot_mat * cam_mat;

    m_cam.cam2world = {
        {new_mat[0][0], new_mat[1][0], new_mat[2][0], new_mat[3][0]},
        {new_mat[0][1], new_mat[1][1], new_mat[2][1], new_mat[3][1]},
        {new_mat[0][2], new_mat[1][2], new_mat[2][2], new_mat[3][2]},
    };
}

glm::vec3 TrackballCamera::getViewDir()
{
    return {m_cam.cam2world.m0.z, m_cam.cam2world.m1.z, m_cam.cam2world.m2.z};
}

glm::vec3 TrackballCamera::getViewUp()
{
    return {m_cam.cam2world.m0.y, m_cam.cam2world.m1.y, m_cam.cam2world.m2.y};
}

glm::vec3 TrackballCamera::getViewSide()
{
    return {m_cam.cam2world.m0.x, m_cam.cam2world.m1.x, m_cam.cam2world.m2.x};
}

glm::vec3 TrackballCamera::getViewPosition()
{
    return {m_cam.cam2world.m0.w, m_cam.cam2world.m1.w, m_cam.cam2world.m2.w};
}

glm::vec3 TrackballCamera::getLookAt()
{
    return getViewPosition() + getViewDir() * m_radius;
}

void TrackballCamera::setLookAt(const glm::vec3& pos)
{
    glm::vec3 dir = pos - getLookAt();
    m_cam.cam2world.m0.w += dir.x;
    m_cam.cam2world.m1.w += dir.y;
    m_cam.cam2world.m2.w += dir.z;
}

void TrackballCamera::rotate(float d_x, float d_y)
{
    d_x *= -PI_2 / m_cam.resolution.x;
    d_y *= PI_2 / m_cam.resolution.y;

    glm::mat4x4 rotation_matrix_x(1.0f);
    glm::mat4x4 rotation_matrix_y(1.0f);

    rotation_matrix_x = glm::rotate(rotation_matrix_x, d_x, m_up);
    rotation_matrix_y = glm::rotate(rotation_matrix_y, d_y, glm::vec3{m_cam.cam2world.m0.x, m_cam.cam2world.m1.x, m_cam.cam2world.m2.x});

	auto rotation_matrix = glm::mat3(rotation_matrix_x * rotation_matrix_y);

    auto old_look_at = getLookAt();
    setLookAt({0.f, 0.f, 0.f});
    addRotation(rotation_matrix);
    setLookAt(old_look_at);

    m_cam.world2cam = m_cam.cam2world.inverse();
}

void TrackballCamera::zoom(float distance)
{
    float scale_factor = std::pow(1.05f, -distance) * m_radius;

    auto prev_look_at = getLookAt();
    auto direction = getViewPosition() - prev_look_at;
    direction *= scale_factor / m_radius;
    direction += prev_look_at;
    m_cam.cam2world.m0.w = direction[0];
    m_cam.cam2world.m1.w = direction[1];
    m_cam.cam2world.m2.w = direction[2];
    m_radius = scale_factor;

    m_cam.world2cam = m_cam.cam2world.inverse();
}

void TrackballCamera::pan(float dx, float dy)
{
    glm::vec3 rel = {dx, -dy, 0.f};
    auto rot = getRotation();
    glm::vec3 movement = rot * rel;

    m_cam.cam2world.m0.w += movement[0];
    m_cam.cam2world.m1.w += movement[1];
    m_cam.cam2world.m2.w += movement[2];

    m_cam.world2cam = m_cam.cam2world.inverse();
}

void TrackballCamera::onUpdate(float delta_time)
{

}

void TrackballCamera::onEvent(IEvent& e)
{

}