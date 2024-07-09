#pragma once

#include <string>
#include <stdexcept>

#define MY_CHECK_THROW(x, str)                                                                                                        \
    do                                                                                                                             \
    {                                                                                                                              \
        if (!(x))                                                                                                 \
            throw std::runtime_error(std::string("HOST | " FILE_LINE " " "failed with error: ") + std::string(str)); \
    } while (0)