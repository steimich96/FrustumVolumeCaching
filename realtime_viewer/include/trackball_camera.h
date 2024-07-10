/*
 * Copyright (C) 2024, Michael Steiner, Graz University of Technology.
 * This code is licensed under the MIT license.
 */
#pragma once

#include "window.h"
#include "events/mouse_events.h"
#include "events/keyboard_events.h"
#include "events/window_events.h"

#include <common.h>
#include <glm/gtc/matrix_transform.hpp>

#include <memory>
#include <numbers>

constexpr float PI_2 = 2.f * M_PI;

class TrackballCamera {
public:
    TrackballCamera(glm::vec3 target, float radius, CameraInfo cam);

    CameraInfo getInfo() { return m_cam; }
    CameraInfo& getInfoRef() { return m_cam; }
    void setCameraMatrix(CameraMatrix matrix) { m_cam = m_cam.createUpdated(matrix, m_cam.resolution); }

    void resize(int2 resolution) { m_cam.resize(resolution); }

    void onUpdate(float delta_time);
    void onEvent(IEvent& e);

    void rotate(float d_x, float d_y);
    void zoom(float distance);
    void pan(float dx, float dy);

private:
    glm::mat3x3 getRotation();
    glm::vec3 getViewDir();
    glm::vec3 getViewUp();
    glm::vec3 getViewSide();
    glm::vec3 getViewPosition();
    glm::vec3 getLookAt();

    void addRotation(const glm::mat3x3& rot_mat);
    void setLookAt(const glm::vec3& pos);

    CameraInfo m_cam;

    float m_theta;
    float m_phi;
    float m_radius;
    glm::vec3 m_up;

    glm::vec3 m_target;
    glm::mat4 m_view;
};