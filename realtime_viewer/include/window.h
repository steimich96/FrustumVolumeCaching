#pragma once

#include "event.h"

#include <interop_buffer.h>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <string>

struct WindowConfig
{
    std::string Title;
    uint32_t Width;
    uint32_t Height;
    bool Resizable;

    explicit WindowConfig(
        const std::string& title, uint32_t width, uint32_t height, bool resizable)
        : Title(title)
        , Width(width)
        , Height(height)
        , Resizable(resizable)
    {}
};

class Window
{
public:
    ~Window() = default;

    Window(const EventCallback& callback, const WindowConfig& config);

    void beginFrame();
    void endFrame();

    void onUpdate();
    void onEvent(IEvent& event);

    void setEventCallback(const EventCallback& callback);

    void resize(uint32_t width, uint32_t heiht);
    void setResizeable(bool resizeable);
    void close();

    bool isMouseButtonpressed(const int button);

    void* getHandle();
    uint32_t getWidth();
    uint32_t getHeight();
    uint32_t getFramebufferWidth();
    uint32_t getFramebufferHeight();
    bool isMinimized();

    void onResize(unsigned int width, unsigned int height);
    void onFramebufferResize(unsigned int width, unsigned int height);

    void triggerEventCallback(IEvent& event);

private:
    uint32_t m_width;
    uint32_t m_height;

    std::string m_title;

    uint32_t m_framebufferWidth;
    uint32_t m_framebufferHeight;

    GLFWwindow* m_handle;

    EventCallback m_eventCallback;

    void setCallbackFunctions();
};
