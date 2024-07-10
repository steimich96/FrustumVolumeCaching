/*
 * Copyright (C) 2024, Michael Steiner, Graz University of Technology.
 * This code is licensed under the MIT license.
 */
#pragma once

#include <cstdint>

namespace CTTI {
    using TypeID = uint64_t;

    template<typename S>
    struct fnv_internal;

    template<>
    struct fnv_internal<uint32_t>
    {
        constexpr static uint32_t default_offset_basis = 2166136261;
        constexpr static uint32_t prime = 16777619;
    };


    template<>
    struct fnv_internal<uint64_t>
    {
        constexpr static uint64_t default_offset_basis = 14695981039346656037ull;
        constexpr static uint64_t prime = 1099511628211ull;
    };

    static constexpr uint32_t hash_str_32(
        char const* const str, const uint32_t val = fnv_internal<uint32_t>::default_offset_basis)
    {
        return (str[0] == '\0') ?
            val :
            hash_str_32(&str[1], (val ^ uint32_t(str[0])) * fnv_internal<uint32_t>::prime);
    }


    static constexpr uint64_t hash_str_64(
        char const* const str, const uint64_t val = fnv_internal<uint64_t>::default_offset_basis)
    {
        return (str[0] == '\0') ?
            val :
            hash_str_64(&str[1], (val ^ uint64_t(str[0])) * fnv_internal<uint64_t>::prime);
    }


    static constexpr TypeID hash_str(const char* input)
    {
        return hash_str_64(input);
    }

    template<typename T>
    static constexpr const char* get_unique_type_name()
    {
    #ifdef _MSC_VER
        return __FUNCSIG__;
    #else
        return __PRETTY_FUNCTION__;
    #endif
    }

    template<class T>
    class TypeIdentifier
    {
    private:
        static inline TypeID m_count = 0;

    public:
        template<class U = T>
        static TypeID generateNewID()
        {
            static const TypeID idCounter = m_count++;
            return idCounter;
        }

        static constexpr TypeID getStaticID()
        {
            // using sanitized = typename std::remove_const_t<std::remove_reference_t<T>>;
            return hash_str(get_unique_type_name<T>());
        }
    };

}
