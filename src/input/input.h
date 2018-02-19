#ifndef INPUT_INPUT_H
#define INPUT_INPUT_H

#include <fstream>
#include <string>

class InputReader
{
    InputReader(const std::string &filename);
    ~InputReader();
};

#endif