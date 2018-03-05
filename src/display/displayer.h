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

struct Vertex
{
	glm::vec3 pos;
	glm::vec3 col;
};

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
	void key_down(SDL_KeyboardEvent& ev);
	void key_up(SDL_KeyboardEvent& ev);
	//void mouse_down(SDL_MouseButtonEvent& ev);
	//void mouse_up(SDL_MouseButtonEvent& ev);
	void mouse_move(SDL_MouseMotionEvent& ev);
	void resize_window(int width, int height);

	int num_points;
	bool quit, pause;
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