#include "directory.h"
#include <algorithm>
#include <string>

namespace
{

struct LexicalCompare
{
    bool operator()(const std::string &lhs, const std::string &rhs)
    {
        if ( lhs.size() < rhs.size() )
        { 
            return true; 
        }
        else if ( rhs.size() < lhs.size() ) 
        {
            return false;
        } 

        //Only reach this point if they are the same length
        return lhs < rhs;
    }

};


}

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

    std::sort( filenames.begin(), filenames.end(), LexicalCompare{} );

    return filenames ;
}