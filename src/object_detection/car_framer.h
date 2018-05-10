#ifndef OBJECT_DETECTION_CAR_FRAMER_H
#define OBJECT_DETECTION_CAR_FRAMER_H

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "../input/input.h"
#include "../colorer/cam_calibration.h"

class CarFramer
{
public:
    CarFramer(int num_of_cams);
    void init(const CamCalibration &cam_calibration);
    void display_frame(const std::vector<Point3D> &points, const std::string folder);

private:
    const int half_width = 350, half_height = 200;

    CamCalibration cam_calibration;
    cv::Mat current_image;
    CameraSelector camera_selector;
};

#endif