#include <math.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>

#include <glm/glm.hpp>

#include "displayer.h"
#include "../util/input.h"
#include "../util/debug.hpp"

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
		std::cerr << "Error while loading shader: " <<  filename << "\n";
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

Displayer::Displayer()
{
	vaoID     = 0;
	vboID     = 0;
	programID = 0;

	is_paused = false;
	is_over   = false;

	prev_tick = SDL_GetTicks();
}


Displayer::~Displayer()
{
}

bool Displayer::init()
{
	glClearColor(0.125f, 0.25f, 0.5f, 1.0f);

	glPointSize(10.0f);

	glEnable(GL_CULL_FACE);
	glEnable(GL_DEPTH_TEST);

	glGenVertexArrays(1, &vaoID);
	glBindVertexArray(vaoID);
	
	glGenBuffers(1, &vboID); 

	InputReader in("../data/2/fusioned_no_color.xyz");
	frame_points = in.get_points();
	num_points = frame_points.size();
	frame_vertices.reserve(num_points);

	std::transform(frame_points.begin(), frame_points.end(), frame_vertices.begin(),
		[](const Point3D &p)
		{
			return Vertex{ glm::vec3{p.x, p.z, p.y}, {1.0, 1.0, 1.0} };
		});
	
	glBindBuffer(GL_ARRAY_BUFFER, vboID);
	glBufferData( GL_ARRAY_BUFFER,
				  num_points * sizeof(Vertex),
				  frame_vertices.data(),
				  GL_STATIC_DRAW);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(
		(GLuint)0,
		3,
		GL_FLOAT,
		GL_FALSE,
		sizeof(Vertex),
		0); 

	glEnableVertexAttribArray(1);
	glVertexAttribPointer(
		(GLuint)1,
		3, 
		GL_FLOAT,
		GL_FALSE,
		sizeof(Vertex),
		(void*)(sizeof(glm::vec3)) );

	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	GLuint vs_ID = load_shader(GL_VERTEX_SHADER,   "display/shaders/myVert.vert");
	GLuint fs_ID = load_shader(GL_FRAGMENT_SHADER, "display/shaders/myFrag.frag");

	programID = glCreateProgram();

	glAttachShader(programID, vs_ID);
	glAttachShader(programID, fs_ID);

	glBindAttribLocation( programID, 0, "vs_in_pos");
	glBindAttribLocation( programID, 1, "vs_in_col");

	glLinkProgram(programID);

	GLint infoLogLength = 0, result = 0;

	glGetProgramiv(programID, GL_LINK_STATUS, &result);
	glGetProgramiv(programID, GL_INFO_LOG_LENGTH, &infoLogLength);
	if (GL_FALSE == result )
	{
		char* error = new char[infoLogLength];
		glGetProgramInfoLog(programID, infoLogLength, NULL, error);
		std::cerr << "[displayer init] Error while creating shader " << error << std::endl;
		delete[] error;
	}

	glDeleteShader( vs_ID );
	glDeleteShader( fs_ID );

	camera.SetProj(45.0f, 640.0f / 480.0f, 0.01f, 1000.0f);

	MVP_loc = glGetUniformLocation(programID, "MVP");

	input_reader.set_path("../data", "fusioned_no_color.xyz");
	input_reader.step();

	return true;
}

void Displayer::clean()
{
	glDeleteBuffers(1, &vboID);
	glDeleteVertexArrays(1, &vaoID);

	glDeleteProgram( programID );
}

void Displayer::update()
{
	float delta = ( SDL_GetTicks() - prev_tick ) / 1000.0f;

	camera.Update(delta);

	prev_tick = SDL_GetTicks();

	MVP = camera.GetViewProj();

	if ( !is_paused && !is_over )
	{
		frame_points = input_reader.next();
		num_points = frame_points.size();
		frame_vertices.reserve(num_points);

		std::transform(frame_points.begin(), frame_points.end(), frame_vertices.begin(),
			[](const Point3D &p)
			{
				return Vertex{ glm::vec3{p.x, p.z, p.y}, {1.0, 1.0, 1.0} };
			});
		
		glBindBuffer(GL_ARRAY_BUFFER, vboID);
		glBufferData( GL_ARRAY_BUFFER,
					num_points * sizeof(Vertex),
					frame_vertices.data(),
					GL_STATIC_DRAW);

		is_over = !input_reader.step();
	}
}


void Displayer::render()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glUseProgram( programID );

	glUniformMatrix4fv( MVP_loc,
						1,
						GL_FALSE,
						&(MVP[0][0]) );

	glBindVertexArray(vaoID);

	glDrawArrays(GL_POINTS, 0, num_points);

	glBindVertexArray(0);

	glUseProgram( 0 );
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
			case SDL_MOUSEMOTION:
				mouse_move(ev.motion);
				break;
			case SDL_WINDOWEVENT:
				if ( ev.window.event == SDL_WINDOWEVENT_SIZE_CHANGED )
				{
					resize_window(ev.window.data1, ev.window.data2);
				}
				break;
			}
}

void Displayer::key_down(SDL_KeyboardEvent& key)
{
	if (key.keysym.sym == SDLK_SPACE)
	{
		is_paused = !is_paused;
	}
	camera.KeyboardDown(key);
}

void Displayer::key_up(SDL_KeyboardEvent& key)
{
	camera.KeyboardUp(key);
}

void Displayer::mouse_move(SDL_MouseMotionEvent& mouse)
{
	camera.MouseMove(mouse);
}

void Displayer::resize_window(int width, int height)
{
	glViewport(0, 0, width, height );

	camera.Resize(width, height);
}