#pragma once

#include "compile_time_type_information.h"

#include <functional>

#define BIT(x) (1 << x)

enum EventCategory
{
    Unknown = 0,
    EventCategoryApplication = BIT(0),
    EventCategoryInput = BIT(1),
    EventCategoryKeyboard = BIT(2),
    EventCategoryMouse = BIT(3),
    EventCategoryMouseButton = BIT(4)
};

// Base Interface for Event. Defines Event behaviour
class IEvent
{
public:
    friend class EventDispatcher;
    friend class Application;

    virtual CTTI::TypeID getID() const = 0;
    virtual bool isInCategory(EventCategory eventCategory) const = 0;

    /*
    template<typename... Cs>
    inline bool isInCategory(Cs&&... categories) const
    {
        return std::ranges::all_of({isInCategory(categories...)}, [](bool i) {
            return i;
        });
    }
    */

    inline bool isHandled() const { return m_handled; }
    inline void setHandled(bool handled) { m_handled = handled; }

protected:
    virtual ~IEvent() = default;

private:
    bool m_handled = false;
};

// CRTP Class for compile time type ID generation. Pass derived class as Template parameter
template<typename T, EventCategory... Cs>
class Event : public IEvent
{
public:
    friend class EventDispatcher;

    inline CTTI::TypeID getID() const final { return id; }

    inline bool isInCategory(EventCategory eventCategory) const final
    {
        return eventCategory & category;
    }

protected:
    ~Event() override = default;

    static constexpr CTTI::TypeID id = CTTI::TypeIdentifier<T>::getStaticID();
    static constexpr int category = (Unknown | ... | Cs);

private:
    Event() = default;
    friend T;
};

using EventCallback = std::function<void(IEvent&)>;