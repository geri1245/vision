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

void Colorer::next_frame()
{
    if(keep_going)
    {
        points = input_reader.next();
    }
}

void Colorer::set_path(
    const std::string &path_,
    const std::string &in_filename_,
    const std::string &out_filename_,
    const std::string &cam_calibration_file_name_)
{
    input_reader.set_path(path_, in_filename_);
    path = path_;
    in_filename  = in_filename_;
    out_filename = out_filename_;
    cam_calibration_file_name = cam_calibration_file_name_;

    in.open(path + "/" + cam_calibration_file_name);
	in >> cam_calibration;
}

void Colorer::print_colors()
{
    out.open(input_reader.get_current_file() + "/" + out_filename);
    std::cout << input_reader.get_current_file() << "\n";
    out << colors;
    out.close();
}

void Colorer::find_colors()
{
    int cam_num;
    while(keep_going)
    {
        read_images();
        next_frame();
        colors.clear();
        colors.reserve( points.size() );
        for(const auto &p : points)
        {
            if(p.x == 0 && p.y == 0 && p.z == 0)
            {
                colors.push_back( {0, 0, 0} );
                continue;
            }

            cam_num = camera_selector(p);
            glm::vec2 coords = cam_calibration.image_calibrations[cam_num].get_pixel_coords(p, 0);
            if(coords.x < 0 || coords.x > 1287 || coords.y < 0 || coords.y > 963)
            {
                colors.push_back( {0, 0, 0} );
            }
            else
            {
                cv::Vec3b col = camera_images[cam_num].at<cv::Vec3b>(clamp(coords.y, 0, 963), clamp(coords.x, 0, 1287));
                colors.push_back(col);
            }
        }
        print_colors();
        keep_going = input_reader.step();
    }
}

int main()
{
    Colorer colorer;
    std::string directory_path, calibration_name, tmp; 
    std::string in_file_name, out_file_name;
    std::ifstream in{ "conf.txt" };
    in >> tmp >>
        directory_path >> tmp >> 
        calibration_name >> tmp >>
        in_file_name >> tmp >>
        out_file_name;

    colorer.set_path(
        directory_path,
        in_file_name,
        out_file_name,
        calibration_name
    );

    colorer.find_colors();

    return 0;
}