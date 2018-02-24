#ifndef VISUALIZATION_H
#define VISUALIZATION_H

// GLEW
#include <GL/glew.h>

// SDL
#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>

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
	void key_down(SDL_KeyboardEvent&);
	void key_up(SDL_KeyboardEvent&);
	void mouse_down(SDL_MouseButtonEvent&);
	void mouse_up(SDL_MouseButtonEvent&);
	void mouse_move(SDL_MouseMotionEvent&);
	void mouse_wheel(SDL_MouseWheelEvent&);
	void resize(int, int);
};

#endif