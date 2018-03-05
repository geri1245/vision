#include <vector>
#include <iostream>

template <typename T>
void print(std::vector<T> vec, std::ostream &out = std::cout)
{
    for(const auto &a : vec)
    {
        out << a;
    }
}