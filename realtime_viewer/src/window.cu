#include "window.h"
#include "core.h"
#include "events/window_events.h"
#include "events/keyboard_events.h"
#include "events/mouse_events.h"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <iostream>

static void GLFWErrorCallback(int error, const char* description)
{
    std::cout << "GLFW Error(" << error << "):" << description << std::endl;
}

Window::Window(const EventCallback& callback, const WindowConfig& config)
    : m_width(config.Width)
    , m_height(config.Height)
    , m_eventCallback(callback)
{
    auto success = glfwInit();
    if (!success)
    {
        const char* description;
        glfwGetError(&description);
        MY_CHECK_THROW(false, "Failed to initialize GLFW");
    }

    //glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);

    //glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    glfwSetErrorCallback(GLFWErrorCallback);
    m_handle = glfwCreateWindow(m_width, m_height, m_title.c_str(), nullptr, nullptr);
    glfwSetWindowAttrib(m_handle, GLFW_RESIZABLE, config.Resizable ? GLFW_TRUE : GLFW_FALSE);
    MY_CHECK_THROW(m_handle, "Failed to create window!");

    int width, height;
    glfwGetFramebufferSize(m_handle, &width, &height);
    m_framebufferWidth = static_cast<uint32_t>(width);
    m_framebufferHeight = static_cast<uint32_t>(height);

    glfwMakeContextCurrent(m_handle);

    MY_CHECK_THROW(gladLoadGLLoader((GLADloadproc)glfwGetProcAddress), "Failed to initialize Glad (OpenGL Loader)!");

    std::cout << "OpenGL Info:" << std::endl;
    std::cout << "  Vendor: " << (const char*)glGetString(GL_VENDOR) << std::endl;
    std::cout << "  Renderer: " << (const char*)glGetString(GL_RENDERER) << std::endl;
    std::cout << "  Version: " << (const char*)glGetString(GL_VERSION) << std::endl;

    glfwSwapInterval(0);
    setCallbackFunctions();

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
    // io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;         // IF using Docking Branch
    // io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(m_handle, true);          // Second param install_callback=true will install GLFW callbacks and chain to existing ones.
    ImGui_ImplOpenGL3_Init();
}

void Window::setCallbackFunctions()
{
    glfwSetWindowUserPointer(m_handle, this);

    glfwSetWindowSizeCallback(m_handle, [](GLFWwindow* glfwWindow, int width, int height) {
        auto* window = (Window*)glfwGetWindowUserPointer(glfwWindow);
        MY_CHECK_THROW(window, "Window instance was not copied glfw user pointer");

        window->onResize(static_cast<uint32_t>(width), static_cast<uint32_t>(height));

        WindowResizeEvent event(width, height);
        window->triggerEventCallback(event);
    });

    glfwSetFramebufferSizeCallback(m_handle, [](GLFWwindow* glfwWindow, int width, int height) {
        auto* window = (Window*)glfwGetWindowUserPointer(glfwWindow);
        MY_CHECK_THROW(window, "Window instance was not copied glfw user pointer");

        window->onFramebufferResize(static_cast<uint32_t>(width), static_cast<uint32_t>(height));

        FramebufferResizeEvent event(width, height);
        window->triggerEventCallback(event);
    });

    glfwSetWindowCloseCallback(m_handle, [](GLFWwindow* glfwWindow) {
        auto* window = (Window*)glfwGetWindowUserPointer(glfwWindow);
        MY_CHECK_THROW(window, "Window instance was not copied glfw user pointer");

        WindowCloseEvent event;
        window->triggerEventCallback(event);
    });

    glfwSetScrollCallback(m_handle, [](GLFWwindow* glfwWindow, double xoffset, double yoffset) {
        auto* window = (Window*)glfwGetWindowUserPointer(glfwWindow);
        MY_CHECK_THROW(window, "Window instance was not copied glfw user pointer");

        MouseScrollEvent event((float)xoffset, (float)yoffset);
        window->triggerEventCallback(event);
    });

    glfwSetWindowRefreshCallback(m_handle, [](GLFWwindow* glfwWindow) {
        auto* window = (Window*)glfwGetWindowUserPointer(glfwWindow);
        MY_CHECK_THROW(window, "Window instance was not copied glfw user pointer");

        WindowRefreshEvent event{};
        window->triggerEventCallback(event);
    });

    glfwSetWindowPosCallback(m_handle, [](GLFWwindow* glfwWindow, int x_pos, int y_pos) {
        auto* window = (Window*)glfwGetWindowUserPointer(glfwWindow);
        MY_CHECK_THROW(window, "Window instance was not copied glfw user pointer");

        WindowMovedEvent event(static_cast<uint32_t>(x_pos), static_cast<uint32_t>(y_pos));
        window->triggerEventCallback(event);
    });

    glfwSetCursorPosCallback(m_handle, [](GLFWwindow* glfwWindow, double x_pos, double y_pos) {
        auto* window = (Window*)glfwGetWindowUserPointer(glfwWindow);
        MY_CHECK_THROW(window, "Window instance was not copied glfw user pointer");

        MouseMovedEvent event(static_cast<float>(x_pos), static_cast<float>(y_pos));
        window->triggerEventCallback(event);
    });

    glfwSetMouseButtonCallback(
        m_handle, [](GLFWwindow* glfwWindow, int button, int action, int /*mods*/) {
            auto* window = (Window*)glfwGetWindowUserPointer(glfwWindow);
            MY_CHECK_THROW(window, "Window instance was not copied glfw user pointer");

            switch (action)
            {
                case GLFW_PRESS: {
                    MouseButtonPressedEvent event((int)button);
                    window->triggerEventCallback(event);
                    break;
                }
                case GLFW_RELEASE: {
                    MouseButtonReleasedEvent event((int)button);
                    window->triggerEventCallback(event);
                    break;
                }
            }
        });

    glfwSetKeyCallback(
        m_handle, [](GLFWwindow* glfwWindow, int key, int /*scancode*/, int action, int /*mods*/) {
            auto* window = (Window*)glfwGetWindowUserPointer(glfwWindow);
            MY_CHECK_THROW(window, "Window instance was not copied glfw user pointer");

            switch (action)
            {
                case GLFW_PRESS: {
                    KeyPressedEvent event((int)key, 0);
                    window->triggerEventCallback(event);
                    break;
                }
                case GLFW_RELEASE: {
                    KeyReleasedEvent event((int)key);
                    window->triggerEventCallback(event);
                    break;
                }
                case GLFW_REPEAT: {
                    KeyPressedEvent event((int)key, 1);
                    window->triggerEventCallback(event);
                    break;
                }
            }
        });

    glfwSetCharCallback(m_handle, [](GLFWwindow* glfwWindow, unsigned int keycode) {
        auto* window = (Window*)glfwGetWindowUserPointer(glfwWindow);
        MY_CHECK_THROW(window, "Window instance was not copied glfw user pointer");

        KeyTypedEvent event((int)keycode);
        window->triggerEventCallback(event);
    });
}

bool Window::isMouseButtonpressed(const int button)
{
    return glfwGetMouseButton(m_handle, button) == GLFW_PRESS;
}

void Window::onResize(unsigned int width, unsigned int height)
{
    m_width = width;
    m_height = height;
}

void Window::onFramebufferResize(unsigned int width, unsigned int height)
{
    m_framebufferWidth = width;
    m_framebufferHeight = height;
}

void Window::onUpdate()
{
    glfwPollEvents();
}

void Window::onEvent(IEvent& e)
{
    // Set event handled if ImGUI interaction to prevent camera movement
    ImGuiIO& io = ImGui::GetIO();

    e.setHandled(e.isHandled() | (e.isInCategory(EventCategoryMouse) & io.WantCaptureMouse));
    e.setHandled(e.isHandled() | (e.isInCategory(EventCategoryKeyboard) & io.WantCaptureKeyboard));
}

void Window::setEventCallback(const EventCallback& callback)
{
    m_eventCallback = callback;
}

void Window::resize(uint32_t width, uint32_t height)
{
    glfwSetWindowSize(m_handle, width, height);
}

void Window::close()
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(m_handle);
    glfwTerminate();
}

void Window::triggerEventCallback(IEvent& event)
{
    m_eventCallback(event);
}

void* Window::getHandle()
{
    return static_cast<void*>(m_handle);
}

uint32_t Window::getWidth()
{
    return m_width;
}
uint32_t Window::getHeight()
{
    return m_height;
}

uint32_t Window::getFramebufferWidth()
{
    return m_framebufferWidth;
}

uint32_t Window::getFramebufferHeight()
{
    return m_framebufferHeight;
}

bool Window::isMinimized()
{
    return glfwGetWindowAttrib(m_handle, GLFW_ICONIFIED);
}

void Window::setResizeable(bool resizable)
{
    glfwSetWindowAttrib(m_handle, GLFW_RESIZABLE, resizable ? GLFW_TRUE : GLFW_FALSE);
}

void Window::beginFrame()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void Window::endFrame()
{
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    if (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
    {
        GLFWwindow* backup_current_context = glfwGetCurrentContext();
        ImGui::UpdatePlatformWindows();
        ImGui::RenderPlatformWindowsDefault();
        glfwMakeContextCurrent(backup_current_context);
    }

    glfwSwapBuffers(m_handle);
}