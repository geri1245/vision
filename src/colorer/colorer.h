#ifndef DISPLAY_COLORER_H
#define DISPLAY_COLORER_H

#include <vector>
#include <string>
#include <fstream>
#include <experimental/filesystem>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <glm/glm.hpp>

#include "../util/input.h"
#include "../util/cam_calibration.h"

struct SelectCamera
{
    int operator()(const Point3D &p)
    {
        Point3D norm_p = normalize(p);
        double x = norm_p.x;
        double z = norm_p.z;
        int cam = 0;
        
        if(x > 0)
        {
            if(x > cosa)
                cam = 2;
            else if(z > 0)
                cam = 4;
            else
                cam = 1;
        }

        else if(x <= 0)
        {
            if(x < -cosa)
                cam = 0;
            else if(z > 0)
                cam = 5;
            else 
                cam = 3;
        }

        return cam;
    }
    float cosa = cos(0.25 * M_PI); //cos(45)
};

class Colorer
{
public:
    void set_path(
        const std::string &path_,
        const std::string &in_filename_,
        const std::string &out_filename_,
        const std::string &cam_calibration_file_path_
        );
    explicit Colorer(int num_of_cams = 6);
        
private:
    void read_images();
    bool next_frame();
    void print_colors();

    int num_of_cams;

    std::string path;
    std::string in_filename;
    std::string out_filename;
    std::string cam_calibration_file_path;

    std::ofstream out;
    std::ifstream in;

    DirInputReader input_reader;
    CamCalibration cam_calibration;
    SelectCamera camera_selector;

    std::vector<cv::Mat> camera_images;
    std::vector<Point3D> points;
    std::vector<glm::vec3> colors;
};

#endif