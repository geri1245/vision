#include "colorer.h"
#include <opencv2/opencv.hpp>

Point3D SelectCamera::normalize(const Point3D &p)
{
    //We don't care about y coordinate
    float square_sum_root = sqrt(p.x * p.x + p.z * p.z);

    return {
        p.x / square_sum_root,
        0,
        p.z / square_sum_root};
}

std::ostream& operator<<(std::ofstream &out, const std::vector<cv::Vec3b> &colors)
{
    for(const auto &color : colors)
    {
        //Opencv uses BGR order, converting it to RGB
        out <<  (int) color[2] << " "  <<
                (int) color[1] << " "  <<
                (int) color[0] << "\n";
    }

    return out;
}

int clamp(int input, int min, int max)
{
    int ret_val = input;
    if(ret_val < min)
        ret_val = min;
    else if(ret_val > max)
        ret_val = max;

    return ret_val;
}

Colorer::Colorer(int num_of_cams) : 
    num_of_cams(num_of_cams),
    cam_calibration(num_of_cams)
{
    camera_images.resize(6);
}

void Colorer::read_images()
{
    std::string file_path = input_reader.get_current_file();

    for(int i = 0; i < num_of_cams; ++i)
    {
        camera_images[i] = 
            cv::imread(
                file_path + "/cam" + std::to_string(i) + ".jpg",
                CV_LOAD_IMAGE_COLOR
            );
    }
}

bool Colorer::next_frame()
{
    bool keep_going = input_reader.step();
    if(keep_going)
    {
        points = input_reader.next();
    }

    return keep_going;
}

void Colorer::set_path(
    const std::string &path_,
    const std::string &in_filename_,
    const std::string &out_filename_,
    const std::string &cam_calibration_file_path_)
{
    input_reader.set_path(path_, in_filename_);
    path = path_;
    in_filename  = in_filename_;
    out_filename = out_filename_;
    cam_calibration_file_path = cam_calibration_file_path_;

    in.open(cam_calibration_file_path);
	in >> cam_calibration;
}

void Colorer::print_colors()
{
    out.open(input_reader.get_current_file() + "/" + out_filename);
    std::cout << input_reader.get_current_file() + "/" + out_filename << "\n";
    out << colors;
    out.close();
}

void Colorer::find_colors()
{
    read_images();
    int cam_num;
    while(next_frame())
    {
        colors.clear();
        colors.reserve( points.size() );
        for(const auto &p : points)
        {
            if(p.x == 0 && p.y == 0 && p.z == 0)
                continue;

            cam_num = camera_selector(p);
            glm::vec2 coords = cam_calibration.image_calibrations[cam_num].get_pixel_coords(p, 0);
            std::cout << cam_num << " " << coords.x << " " << coords.y << "\n";
            cv::Vec3b col = camera_images[cam_num].at<cv::Vec3b>(clamp(coords.y, 0, 963), clamp(coords.x, 0, 1287));
            colors.push_back(col);
        }
        print_colors();
    }
}

int main()
{
    Colorer colorer;
    colorer.set_path(
        "../../data1",
        "lidar1.xyz",
        "lidar1.col",
        "../../data1/calibration.txt"
    );
    colorer.find_colors();

    return 0;
}