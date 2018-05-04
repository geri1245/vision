#ifndef DISPLAY_COLORER_H
#define DISPLAY_COLORER_H

#include <vector>
#include <string>
#include <fstream>
#include <experimental/filesystem>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <glm/glm.hpp>

#include "../input/input.h"
#include "cam_calibration.h"

class Colorer
{
public:
    void set_path(
        const std::string &path_,
        const std::string &in_filename_,
        const std::string &out_filename_,
        const std::string &cam_calibration_file_name_
        );
    explicit Colorer(int num_of_cams = 6);

    void find_colors();
        
private:
    void read_images();
    void next_frame();
    void print_colors();

    int num_of_cams;
    bool keep_going = true;

    std::string path;
    std::string in_filename;
    std::string out_filename;
    std::string cam_calibration_file_name;

    std::ofstream out;
    std::ifstream in;

    DirInputReader input_reader;
    CamCalibration cam_calibration;
    CameraSelector camera_selector;

    std::vector<cv::Mat> camera_images;
    std::vector<Point3D> points;
    std::vector<cv::Vec3b> colors;
};

#endif