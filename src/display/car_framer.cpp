#include "car_framer.h"

namespace
{
    int clamp(int value, int min, int max)
    {
        if(value < min)
            return min;

        if(value > max)
            return max;

        return value;
    }
}

CarFramer::CarFramer(int num_of_cams) :
    cam_calibration(num_of_cams)
{}

void CarFramer::init(const CamCalibration &cam_calibration)
{
    this->cam_calibration = cam_calibration;
}

void CarFramer::display_frame(const std::vector<Point3D> &points, const std::string folder)
{
    cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );

    for(const auto &p : points)
    {
        int cam_num = camera_selector(p);

        current_image = cv::imread(folder + "/cam" + std::to_string(cam_num) + ".jpg");

        glm::vec2 coords = cam_calibration.image_calibrations[cam_num].get_pixel_coords(p, 0);
        
        if(coords.x > 0 && coords.x < 1287 && coords.y > 0 && coords.y < 963)
        {
            int x_start = clamp(coords.x - half_width, 0, 1287);
            int x_end   = clamp(coords.x + half_width, 0, 1287);
            int y_start = clamp(coords.y - half_height, 0, 963);
            int y_end   = clamp(coords.y + half_height, 0, 963);

            for(int i = x_start; i < x_end; ++i)
            {
                current_image.at<cv::Vec3b>(y_start, i) = cv::Vec3b{0, 0, 255};
                current_image.at<cv::Vec3b>(y_end, i) = cv::Vec3b{0, 0, 255};
            }
            for(int j = y_start; j < y_end; ++j)
            {
                current_image.at<cv::Vec3b>(j, x_start) = cv::Vec3b{0, 0, 255};
                current_image.at<cv::Vec3b>(j, x_end) = cv::Vec3b{0, 0, 255};
            }
        }
        cv::imshow( "Display window", current_image );

        cv::waitKey(0); 
    }
    cv::destroyWindow("Display window");
}