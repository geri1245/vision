#ifndef VISUALIZATION_H
#define VISUALIZATION_H

#include <vector>
#include <array>

#include <GL/glew.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>
#include <glm/glm.hpp>

#include "../util/point.h"
#include "../util/input.h"
#include "gCamera.h"

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
	const std::string in_files_path    = "../data";
	const std::string in_files_name    = "fusioned_no_color.xyz";

	void set_ogl();
	void next_frame();
	
//Events:	
	void key_down(SDL_KeyboardEvent& ev);
	void key_up(SDL_KeyboardEvent& ev);
	void mouse_move(SDL_MouseMotionEvent& ev);
	void resize_window(int width, int height);

	int num_points;
	bool is_over, is_paused;
	Uint32 prev_tick;

	std::vector<Point3D> frame_points;
	std::vector<Vertex>  frame_vertices;

	GLuint programID;

	GLuint vaoID;
	GLuint vboID;

	GLuint MVP_loc;
	glm::mat4 MVP;

	DirInputReader input_reader;
	gCamera camera;

	int mouse_x;
	int mouse_y;
};

#endif