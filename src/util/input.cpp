#include "input.h"
#include "directory.h"

InputReader::InputReader(const std::string &filename_, int size_) : 
    size(size_),
    filename(filename_)
    {}

std::vector<Point3D> InputReader::get_points()
{
    in.open(filename);

    std::vector<Point3D> ret;
    ret.reserve(size);
    
    Point3D p;

    while(in >> p)
    {
        ret.push_back(p);
    }

    return ret;
}

void InputReader::set_filename(const std::string &filename_)
{
    filename = filename_;
}


namespace { namespace fs = std::experimental::filesystem; }

DirInputReader::DirInputReader(const fs::path &path_) : 
    path(path_)
{
    files = files_in_directory(path);
    size = files.size();
}

void DirInputReader::set_path(const std::experimental::filesystem::path &path_)
{
    path = path_;
    current = 0;
    files = files_in_directory(path);
}

bool DirInputReader::step()
{
    ++current;
    if(current >= size)
    {
        return false;
    }

    return true;
}

std::vector<Point3D> DirInputReader::next()
{
    InputReader ir{};
    ir.set_filename( files[current] );
    return ir.get_points();
}