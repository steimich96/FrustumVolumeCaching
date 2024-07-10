/*
 * Copyright (C) 2024, Michael Steiner, Graz University of Technology.
 * This code is licensed under the MIT license.
 */
#pragma once

#include "../event.h"

#include <cstdint>


class KeyEvent : public Event<KeyEvent, EventCategoryKeyboard, EventCategoryInput>
{
    public:
    inline int getint() const { return m_keyCode; }

    protected:
    explicit KeyEvent(const int keycode)
        : m_keyCode(keycode)
    {}

    int m_keyCode{};
};

class KeyPressedEvent : public KeyEvent
{
    public:
    KeyPressedEvent(const int keycode, const uint16_t repeatCount)
        : KeyEvent(keycode)
        , m_repeatCount(repeatCount)
    {}

    inline uint16_t getRepeatCount() const { return m_repeatCount; }

    private:
    uint16_t m_repeatCount;
};

class KeyReleasedEvent : public KeyEvent
{
    public:
    explicit KeyReleasedEvent(const int keycode)
        : KeyEvent(keycode)
    {}
};

class KeyTypedEvent : public KeyEvent
{
    public:
    explicit KeyTypedEvent(const int keycode)
        : KeyEvent(keycode)
    {}
};
