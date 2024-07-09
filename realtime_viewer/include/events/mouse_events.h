#pragma once

#include "../event.h"

class MouseMovedEvent : public Event<MouseMovedEvent, EventCategoryMouse, EventCategoryInput>
{
    public:
    MouseMovedEvent(const float x, const float y)
        : m_mouseX(x)
        , m_mouseY(y)
    {}

    inline float getX() const { return m_mouseX; }
    inline float getY() const { return m_mouseY; }

    private:
    float m_mouseX, m_mouseY;
};

class MouseScrollEvent : public Event<MouseScrollEvent, EventCategoryMouse, EventCategoryInput>
{
    public:
    MouseScrollEvent(float x_offset, float y_offset)
        : m_xoffset(x_offset)
        , m_yoffset(y_offset)
    {}

    inline float getXOffset() const { return m_xoffset; }

    inline float getYOffset() const { return m_yoffset; }

    private:
    float m_xoffset, m_yoffset;
};

template <typename T>
class MouseButtonEvent
    : public Event<MouseButtonEvent<T>, EventCategoryMouse, EventCategoryMouseButton,
            EventCategoryInput>
{
    public:
    explicit MouseButtonEvent(const int button)
        : m_button(button)
    {}

    inline int getMouseButton() const { return m_button; }

    private:
    int m_button;
};

class MouseButtonPressedEvent : public MouseButtonEvent<MouseButtonPressedEvent>
{
    public:
    explicit MouseButtonPressedEvent(const int button)
        : MouseButtonEvent(button)
    {}
};

class MouseButtonReleasedEvent : public MouseButtonEvent<MouseButtonReleasedEvent>
{
    public:
    explicit MouseButtonReleasedEvent(const int button)
        : MouseButtonEvent(button)
    {}
};
