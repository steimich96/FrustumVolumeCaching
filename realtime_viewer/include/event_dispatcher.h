#pragma once

#include "event.h"

#include "events/mouse_events.h"

#define BIND_EVENT_CALLBACK(callback)                                 \
    [this](auto&&... args) -> decltype(auto) {                        \
        return this->callback(std::forward<decltype(args)>(args)...); \
    }


class EventDispatcher
{
    public:
    explicit EventDispatcher(IEvent& event)
        : m_event(event)
    {}

    template<typename T, typename F>
    bool dispatch(const F& func)
    {
        if (T::id == m_event.getID())
        {
            T& cast = static_cast<T&>(m_event);

            m_event.m_handled = func(cast);
            return true;
        }
        return false;
    }

    private:
    IEvent& m_event;
};
