#include "colorer.h"

int main()
{
    Colorer colorer;
    colorer.set_path("conf.txt");

    colorer.find_colors();

    return 0;
}