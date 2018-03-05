#include "directory.h"
#include <algorithm>

std::vector<std::string> files_in_directory(
    const std::experimental::filesystem::path &path)
{
    namespace fs = std::experimental::filesystem;
    std::vector<std::string> filenames ;
    
    const fs::directory_iterator end{} ;
    
    for( fs::directory_iterator it{ path } ; it != end ; ++it )
    {
            filenames.push_back( it->path().string() ) ;
    }

    std::sort( filenames.begin(), filenames.end() );

    return filenames ;
}