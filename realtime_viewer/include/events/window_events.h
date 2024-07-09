#pragma once

#include "../event.h"

#include <cstdint>

class WindowResizeEvent : public Event<WindowResizeEvent, EventCategoryApplication>
{
    public:
    WindowResizeEvent(unsigned int width, unsigned int height)
        : m_width(width)
        , m_height(height)
    {}

    inline unsigned int getWidth() const { return m_width; }

    inline unsigned int getHeight() const { return m_height; }

    private:
    unsigned int m_width, m_height;
};

class FramebufferResizeEvent : public Event<FramebufferResizeEvent, EventCategoryApplication>
{
    public:
    FramebufferResizeEvent(unsigned int width, unsigned int height)
        : m_width(width)
        , m_height(height)
    {}

    inline unsigned int getWidth() const { return m_width; }

    inline unsigned int getHeight() const { return m_height; }

    private:
    unsigned int m_width, m_height;
};

class WindowMovedEvent : public Event<WindowMovedEvent, EventCategoryApplication>
{
    public:
    WindowMovedEvent(uint32_t x_pos, uint32_t y_pos)
        : m_xPos(x_pos)
        , m_yPos(y_pos)
    {}

    inline uint32_t getXPos() const { return m_xPos; }

    inline uint32_t getYPos() const { return m_yPos; }

    private:
    uint32_t m_xPos, m_yPos;
};

class WindowRefreshEvent : public Event<WindowRefreshEvent, EventCategoryApplication>
{
    public:
    WindowRefreshEvent() = default;
};

class WindowCloseEvent : public Event<WindowCloseEvent, EventCategoryApplication>
{
    public:
    WindowCloseEvent() = default;
};
