#ifndef COMMON_H
#define COMMON_H

#include <sstream>
#include <iostream>

namespace Common
{
    inline std::string to_string(int i)
    {
        std::ostringstream ss;
        ss << i;
        return ss.str();
    }
}
#endif // COMMON_H
