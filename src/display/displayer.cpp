#include <math.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include "displayer.h"
#include "../util/input.h"
#include "../util/debug.hpp"

//Predicate for Point3D comparison
struct Pred
{
	bool operator()(const Point3D &lhs, const Point3D &rhs)
	{
		if( lhs.x < rhs.x )
		{
			return true;
		}
		else if ( lhs.x > rhs.x )
		{
			return false;
		}
		else
		{
			return lhs.z < rhs.z;
		}
	}
};

Displayer::Displayer() : camera_images(num_of_cams), cam_calibration(num_of_cams)
{
	vaoID     = 0;
	vboID     = 0;

	is_paused = false;
	is_over   = false;

	prev_tick = SDL_GetTicks();
}


Displayer::~Displayer()
{
}

void Displayer::set_ogl()
{
	//glClearColor(0.125f, 0.25f, 0.5f, 1.0f);
	glClearColor(0, 0, 0, 1.0f);
	glPointSize(point_size);

	//glEnable(GL_POINT_SMOOTH); 

	//glEnable(GL_BLEND);	
	//glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_CULL_FACE);
	glEnable(GL_DEPTH_TEST);
}

void Displayer::next_frame()
{
	frame_points = input_reader.next(camera_images);
	
	std::sort(frame_points.begin(), frame_points.end(), Pred());
	
	num_points = frame_points.size();

	if(num_points != 0) //We only change the displayed points if the frame is not empty
	{
		frame_vertices.clear();

		for( const auto& p : frame_points )
		{
			frame_vertices.push_back( Vertex{ glm::vec3{p.x, p.y, p.z}, {1.0, 1.0, 1.0} } );
		}
	}
}

bool Displayer::init()
{
	set_ogl();
	input_reader.set_path(in_files_path, in_files_name);

	std::ifstream in{cam_calibration_file_path};
	in >> cam_calibration;

	//Construct points for a cube
	std::vector<Vertex> cube_vertices;
	cube_vertices.resize(8);
	cube_vertices.push_back( { glm::vec3(-0.5, -0.5, -0.5), glm::vec3(1, 0, 0) } );
	cube_vertices.push_back( { glm::vec3(-0.5, -0.5,  0.5), glm::vec3(1, 0, 0) } );
	cube_vertices.push_back( { glm::vec3( 0.5, -0.5,  0.5), glm::vec3(1, 0, 0) } );
	cube_vertices.push_back( { glm::vec3( 0.5, -0.5, -0.5), glm::vec3(1, 0, 0) } );
	cube_vertices.push_back( { glm::vec3(-0.5,  0.5, -0.5), glm::vec3(1, 0, 0) } );
	cube_vertices.push_back( { glm::vec3(-0.5,  0.5,  0.5), glm::vec3(1, 0, 0) } );
	cube_vertices.push_back( { glm::vec3( 0.5,  0.5,  0.5), glm::vec3(1, 0, 0) } );
	cube_vertices.push_back( { glm::vec3( 0.5,  0.5, -0.5), glm::vec3(1, 0, 0) } );

	GLushort indices[]=
    {
        0, 1, 2, 2, 3, 0,
		6, 2, 3, 3, 7, 6,
		7, 3, 0, 0, 4, 7,
		4, 0, 2, 2, 5, 4,
		5, 1, 2, 2, 6, 5,
		4, 5, 6, 6, 7, 0
    };

	next_frame();

	program.generate_vao_vbo<Vertex>(vaoID, vboID, num_points * sizeof(Vertex), frame_vertices.data());
	program.generate_vao_vbo<Vertex>(cube_vaoID, cube_vboID, 8 * sizeof(Vertex), cube_vertices.data(), GL_STATIC_DRAW);
	
	glBindVertexArray(cube_vaoID);
	glBindBuffer(GL_ARRAY_BUFFER, cube_vboID);

	glGenBuffers(1, &cube_indexBufferID);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, cube_indexBufferID);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

	program.create_program_with_shaders(
		vert_shader_path, 
		frag_shader_path);

	program.get_uniform(MVP_loc, "MVP");

	return true;
}

void Displayer::clean()
{
	program.clean(vaoID, vboID);
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
		points_to_draw = num_points == 0 ? frame_vertices.size() : num_points;

		program.update_vbo<Vertex>(vboID, points_to_draw * sizeof(Vertex), frame_vertices.data());

		is_over = !input_reader.step();
	}
}


void Displayer::render()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	program.draw_points(vaoID, MVP_loc, MVP, points_to_draw);

	glUseProgram( program.program_id() );
	MVP = MVP * glm::scale(glm::vec3(10.0f));
	glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, &(MVP[0][0]));

	glBindVertexArray(cube_vaoID);
	glDrawElements(GL_TRIANGLES, 6 * 4, GL_UNSIGNED_SHORT, 0);
	glBindVertexArray(0);
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