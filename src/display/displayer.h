#ifndef DISPLAY_DISPLAYER_H
#define DISPLAY_DISPLAYER_H

#include <vector>
#include <array>

#include <GL/glew.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>
#include <glm/glm.hpp>

#include "../util/point.h"
#include "../util/input.h"
#include "gCamera.h"
#include "program.hpp"
#include "color.h"

class Displayer
{
public:
	Displayer();
	~Displayer();

	bool init();
	void clean();

	void update();
	void render();

	void handle_event(SDL_Event);

private:

//Constants
	const std::string vert_shader_path = "display/shaders/myVert.vert";
	const std::string frag_shader_path = "display/shaders/myFrag.frag";
	
	std::string in_files_path;
	std::string in_files_name;
	std::string in_color_file_name;
	int num_of_cams = 6;

	const GLfloat point_size = 8.0f;

	void set_ogl();
	void next_frame();
	
//Events:	
	void key_down(SDL_KeyboardEvent& ev);
	void key_up(SDL_KeyboardEvent& ev);
	void mouse_move(SDL_MouseMotionEvent& ev);
	void resize_window(int width, int height);

	void read_conf_file();
	void init_cube();
	void init_rectangle();

	void draw_cube(const glm::mat4 &world_transform);
	void draw_rectangle(const glm::mat4 &world_transform);
	void draw_points(const glm::mat4 &world_transform);

	void read_colors();

	int num_points, points_to_draw, frame_num = 0;
	float alpha;
	bool is_over, is_paused;
	bool display_colors = false, display_planes = false, display_cars = false;
	Uint32 prev_tick;

	std::vector<Point3D> frame_points;
	std::vector<Point3D> car_points;
	std::vector<Vertex>  frame_vertices;
	std::vector<Color>   colors;
	std::vector < std::vector<Point3D> > planes;

	GLuint vaoID, vboID;
	GLuint cube_vaoID, cube_vboID, cube_indexBufferID;
	GLuint rectangle_vaoID, rectangle_vboID;

	GLuint MVP_loc, alpha_loc;
	glm::mat4 MVP;

	Program program;
	DirInputReader input_reader;
	gCamera camera;

	int mouse_x;
	int mouse_y;
};

#endif