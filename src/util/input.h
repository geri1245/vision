#ifndef INPUT_INPUT_H
#define INPUT_INPUT_H

#include <fstream>
#include <string>
#include <vector>
#include <experimental/filesystem>

#include <GL/glew.h>

#include "point.h"

class InputReader
{
public:

    InputReader()  = default;
    ~InputReader() = default;
    InputReader(const std::string &filename_, int size = 40000);
    
    std::vector<Point3D> get_points();
    void set_filename(const std::string &filename_);
    void set_texture_name(const std::string &texture_name_);

private:

    int size = 40000;
    std::string filename;
    std::string texture_name;
    std::fstream in;
};

class DirInputReader
{
public:

    DirInputReader()  = default;
    ~DirInputReader() = default;
    DirInputReader(
        const std::experimental::filesystem::path &path_,
        const std::string &filename_);

    void set_path(
        const std::experimental::filesystem::path &path_,
        const std::string &filename_);
    
    bool step();
    std::vector<Point3D> next();

    std::string get_current_file();

private:

    int size;
    int current = 0;

    InputReader input_reader;

    std::experimental::filesystem::path path;
    std::string filename;
    std::fstream in;
    std::vector<std::string> files;
};

#endif