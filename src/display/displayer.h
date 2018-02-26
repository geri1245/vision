#ifndef VISUALIZATION_H
#define VISUALIZATION_H

#include <vector>
#include <array>

#include <GL/glew.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>
#include <glm/glm.hpp>

#include "../repr/point.h"

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
	void mouse_down(SDL_MouseButtonEvent& ev);
	void mouse_move(SDL_MouseMotionEvent& ev);
	void resize_window(int width, int height);

	int num_points;

	std::vector<Point3D> frame_points;
	std::vector<Vertex>  frame_vertices;

	GLuint programID;

	GLuint vaoID;
	GLuint vboID;

	int mouse_x;
	int mouse_y;
};

#endif