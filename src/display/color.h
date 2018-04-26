#ifndef DISPLAY_COLOR_H
#define DISPLAY_COLOR_H

#include <stdint.h>
#include <fstream>

struct Color
{
    int r, g, b;
};

std::istream& operator>>(std::istream &in, Color &col);
std::ostream& operator<<(std::ostream &out, const Color &col);

#endif