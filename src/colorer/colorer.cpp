#include "colorer.h"

namespace
{

Point3D normalize(const Point3D &p)
{
    //We don't care about y coordinate
    double square_sum_root = sqrt(p.x * p.x + p.z * p.z);

    return {
        p.x / square_sum_root,
        0,
        p.z / square_sum_root};
}


std::ostream& operator<<(std::ofstream &out, const std::vector<glm::vec3> &colors)
{
    for(const auto &color : colors)
    {
        out <<  color.x << " "  <<
                color.y << " "  <<
                color.z << "\n";
    }
}

}

Colorer::Colorer(int num_of_cams) : 
    cam_calibration(num_of_cams),
    num_of_cams(num_of_cams)
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

    std::ifstream in{cam_calibration_file_path};
	in >> cam_calibration;
}

void Colorer::print_colors()
{
    out.open(input_reader.get_current_file() + "/" + out_filename);
    out << colors;
    out.close();
}

int main()
{
    
}