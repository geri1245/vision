#include <math.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>

#include <glm/glm.hpp>

#include "displayer.h"
#include "program.hpp"
#include "../util/input.h"
#include "../util/debug.hpp"


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

void Displayer::set_ogl()
{
	glClearColor(0.125f, 0.25f, 0.5f, 1.0f);
	glPointSize(10.0f);

	glEnable(GL_CULL_FACE);
	glEnable(GL_DEPTH_TEST);
}

void Displayer::next_frame()
{
	frame_points = input_reader.next();
	num_points = frame_points.size();
	frame_vertices.reserve(num_points);

	std::transform(frame_points.begin(), frame_points.end(), frame_vertices.begin(),
		[](const Point3D &p)
		{
			return Vertex{ glm::vec3{p.x, p.z, p.y}, {1.0, 1.0, 1.0} };
		});
}

bool Displayer::init()
{
	set_ogl();
	input_reader.set_path(in_files_path, in_files_name);

	next_frame();

	glGenVertexArrays(1, &vaoID);
	glBindVertexArray(vaoID);
	
	glGenBuffers(1, &vboID); 

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

	GLuint vs_ID = load_shader(GL_VERTEX_SHADER,   vert_shader_path);
	GLuint fs_ID = load_shader(GL_FRAGMENT_SHADER, frag_shader_path);

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
		next_frame();
		
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