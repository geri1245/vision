#include <math.h>
#include <vector>
#include <iostream>
#include <fstream>

#include "displayer.h"

namespace
{

GLuint load_shader(GLenum shader_type, const std::string &filename)
{
	GLuint shader = glCreateShader( shader_type );
	
	if ( shader == 0 )
	{
		std::cerr << "Error while initializing shader: " << filename << "\n";
		return 0;
	}
	
	std::string shaderCode = "";
	std::ifstream shaderStream(filename);

	if ( !shaderStream.is_open() )
	{
		std::cerr << "Error while loading shader: " <<  filename;
		return 0;
	}


	std::string next_line = "";
	while ( std::getline(shaderStream, next_line) )
	{
		shaderCode += next_line + "\n";
	}

	shaderStream.close();


	const char* sourcePointer = shaderCode.c_str();
	glShaderSource( shader, 1, &sourcePointer, NULL );
	glCompileShader( shader );

	//Check compilation result
	GLint result = GL_FALSE;
    int infoLogLength;
	
	glGetShaderiv(shader, GL_COMPILE_STATUS, &result);
	glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLogLength);

	if ( GL_FALSE == result )
	{
		std::vector<char> VertexShaderErrorMessage(infoLogLength);
		glGetShaderInfoLog(shader, infoLogLength, NULL, &VertexShaderErrorMessage[0]);

		std::cerr << VertexShaderErrorMessage[0];
	}

	return shader;
}

}

Displayer::Displayer(void)
{
}


Displayer::~Displayer(void)
{
}

bool Displayer::init()
{
	glClearColor(0.125f, 0.25f, 0.5f, 1.0f);

	glEnable(GL_CULL_FACE);
	glEnable(GL_DEPTH_TEST);

	return true;
}

void Displayer::clean()
{
}

void Displayer::update()
{
}


void Displayer::render()
{
	// t�r�lj�k a frampuffert (GL_COLOR_BUFFER_BIT) �s a m�lys�gi Z puffert (GL_DEPTH_BUFFER_BIT)
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

//Event handling:

void Displayer::handle_event(SDL_Event ev)
{
	switch (ev.type)
			{
			case SDL_KEYDOWN:
				key_down(ev.key);
				break;
			case SDL_KEYUP:
				key_up(ev.key);
				break;
			case SDL_MOUSEBUTTONDOWN:
				mouse_down(ev.button);
				break;
			case SDL_MOUSEBUTTONUP:
				mouse_up(ev.button);
				break;
			case SDL_MOUSEMOTION:
				mouse_move(ev.motion);
				break;
			case SDL_MOUSEWHEEL:
				mouse_wheel(ev.wheel);
				break;
			case SDL_WINDOWEVENT:
				if ( ev.window.event == SDL_WINDOWEVENT_SIZE_CHANGED )
				{
					resize(ev.window.data1, ev.window.data2);
				}
				break;
			}
}

void Displayer::key_down(SDL_KeyboardEvent& key)
{
}

void Displayer::key_up(SDL_KeyboardEvent& key)
{
}

void Displayer::mouse_down(SDL_MouseButtonEvent& mouse)
{
}

void Displayer::mouse_up(SDL_MouseButtonEvent& mouse)
{
}

void Displayer::mouse_move(SDL_MouseMotionEvent& mouse)
{

}

void Displayer::mouse_wheel(SDL_MouseWheelEvent& wheel)
{
}

// a k�t param�terbe az �j ablakm�ret sz�less�ge (_w) �s magass�ga (_h) tal�lhat�
void Displayer::resize(int _w, int _h)
{
	glViewport(0, 0, _w, _h );
}