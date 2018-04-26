#include "color.h"

std::istream& operator>>(std::istream &in, Color &col)
{
    return in >> col.r >> col.g >> col.b;
}

std::ostream& operator<<(std::ostream &out, const Color &col)
{
    return out << 
        "R: "    << col.r <<
        " , G: " << col.g <<
        " , B: " << col.b << "\n";
}