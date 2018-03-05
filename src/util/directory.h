#ifndef UTIL_DIRECTORY_H
#define UTIL_DIRECTORY_H

#include <vector>
#include <string>
#include <experimental/filesystem>

std::vector<std::string> files_in_directory(
    const std::experimental::filesystem::path &path);

#endif