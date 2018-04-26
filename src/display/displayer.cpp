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
#include "../gpu/cpu_ransac_prep.h"


Displayer::Displayer()
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
	glClearColor(0, 0, 0, 1.0f);
	glPointSize(point_size);

	glEnable(GL_POINT_SMOOTH); 

	glEnable(GL_BLEND);	
	glBlendFunc(GL_SRC_ALPHA, GL_ONE);
	//glEnable(GL_DEPTH_TEST);
}

void Displayer::init_cube()
{
	//Construct points for a cube
	std::vector<Vertex> cube_vertices;
	cube_vertices.reserve(8);
	
	cube_vertices.push_back( Vertex{ glm::vec3( 0.5f, -0.5f,  0.5f), glm::vec3(0.3f, 0.f, 0.f) } );
	cube_vertices.push_back( Vertex{ glm::vec3( 0.5f, -0.5f, -0.5f), glm::vec3(0.3f, 0.f, 0.f) } );
	cube_vertices.push_back( Vertex{ glm::vec3(-0.5f, -0.5f, -0.5f), glm::vec3(0.3f, 0.f, 0.f) } );
	cube_vertices.push_back( Vertex{ glm::vec3(-0.5f, -0.5f,  0.5f), glm::vec3(0.3f, 0.f, 0.f) } );
	cube_vertices.push_back( Vertex{ glm::vec3( 0.5f,  0.5f,  0.5f), glm::vec3(0.3f, 0.f, 0.f) } );
	cube_vertices.push_back( Vertex{ glm::vec3( 0.5f,  0.5f, -0.5f), glm::vec3(0.3f, 0.f, 0.f) } );
	cube_vertices.push_back( Vertex{ glm::vec3(-0.5f,  0.5f, -0.5f), glm::vec3(0.3f, 0.f, 0.f) } );
	cube_vertices.push_back( Vertex{ glm::vec3(-0.5f,  0.5f,  0.5f), glm::vec3(0.3f, 0.f, 0.f) } );

	//Cube index buffer
	GLushort cube_indices[]=
    {
        0, 1, 2, 2, 3, 0, //Bottom
		6, 2, 3, 3, 7, 6,
		7, 3, 0, 0, 4, 7,
		4, 0, 1, 1, 5, 4,
		5, 1, 2, 2, 6, 5,
		4, 5, 6, 6, 7, 4
    };

	program.generate_vao_vbo<Vertex>(cube_vaoID, cube_vboID, 8 * sizeof(Vertex), cube_vertices.data(), GL_STATIC_DRAW);

	glBindVertexArray(cube_vaoID);
	glGenBuffers(1, &cube_indexBufferID);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, cube_indexBufferID);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(cube_indices), cube_indices, GL_STATIC_DRAW);

	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void Displayer::init_rectangle()
{
	std::vector<Vertex> rectangle_vertices;
	rectangle_vertices.reserve(4);
	
	rectangle_vertices.push_back( Vertex{ glm::vec3(-0.5f, -0.5f, 0), glm::vec3(0.3f, 0.f, 0.7f) } );
	rectangle_vertices.push_back( Vertex{ glm::vec3( 0.5f, -0.5f, 0), glm::vec3(0.3f, 0.f, 0.7f) } );
	rectangle_vertices.push_back( Vertex{ glm::vec3( 0.5f,  0.5f, 0), glm::vec3(0.3f, 0.f, 0.7f) } );
	rectangle_vertices.push_back( Vertex{ glm::vec3(-0.5f,  0.5f, 0), glm::vec3(0.3f, 0.f, 0.7f) } );

	program.generate_vao_vbo<Vertex>(
		rectangle_vaoID, 
		rectangle_vboID, 
		4 * sizeof(Vertex), 
		rectangle_vertices.data(), GL_STATIC_DRAW);

	glBindVertexArray(0);
}

void Displayer::next_frame()
{
	frame_points = input_reader.next(); //Read points
	num_points = frame_points.size();

	read_colors();
	std::cout << "points: " << num_points << "  colors: " << colors.size() << "\n";
	for(int i = 0; i < 10; ++i)
	{
		std::cout << colors[i] << " ";
	}

	if(num_points != 0) //We only change the displayed points if the frame is not empty
	{
		//Detecting planes
		/*std::vector < std::vector<int> > planes = find_plane(frame_points, 1000, 0.002, 60);
		for(const auto &v : planes)
		{
			for(int n : v)
			{
			frame_vertices[n].col = {1.0, 0, 0};
			}
		}*/
		
		frame_vertices.clear();

		for( int i = 0; i < num_points; ++i )
		{
			const Point3D &p = frame_points[i];
			const Color &c   = colors[i];
			frame_vertices.push_back(
				Vertex{ 
					{p.x, p.y, p.z}, 
					{c.r / 255.f, c.g / 255.f, c.b / 255.f} 
				}
			);
		}
	}
}

void Displayer::read_colors()
{
	std::ifstream in{ input_reader.get_current_file() + "/" + in_color_file_name };
	colors.resize(num_points);
	colors.clear();
	for(int i = 0; i < num_points; ++i)
	{
		Color col;
		in >> col;
		colors.push_back(col);
	}
}

bool Displayer::init()
{
	set_ogl();
	init_cube();
	init_rectangle();

	input_reader.set_path(in_files_path, in_files_name);

	next_frame();

	program.generate_vao_vbo<Vertex>(vaoID, vboID, num_points * sizeof(Vertex), frame_vertices.data());
	
	program.create_program_with_shaders(
		vert_shader_path, 
		frag_shader_path);

	program.get_uniform(MVP_loc, "MVP");
	program.get_uniform(alpha_loc, "alpha");

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
		points_to_draw = num_points == 0 ? frame_vertices.size() : num_points;
		program.update_vbo<Vertex>(vboID, points_to_draw * sizeof(Vertex), frame_vertices.data());

		next_frame();
		is_over = !input_reader.step();
	}
}

void Displayer::draw_cube(const glm::mat4 &world_transform)
{
	alpha = 0.1f;
	glUniform1f(alpha_loc, alpha);

	glUseProgram( program.program_id() );
	MVP = camera.GetViewProj() * world_transform;
	glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, &(MVP[0][0]));

	glBindVertexArray(cube_vaoID);
	glDrawElements(GL_TRIANGLES, 6 * 6, GL_UNSIGNED_SHORT, 0);
	glBindVertexArray(0);
}

void Displayer::draw_rectangle(const glm::mat4 &world_transform)
{
	alpha = 0.7f;
	glUniform1f(alpha_loc, alpha);

	glUseProgram( program.program_id() );
	MVP = camera.GetViewProj() * world_transform;
	glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, &(MVP[0][0]));

	glBindVertexArray(rectangle_vaoID);
	glDrawArrays(GL_QUADS, 0, 4);
	glBindVertexArray(0);
}


void Displayer::render()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	alpha = 1.0f;
	glUniform1f(alpha_loc, alpha);
	program.draw_points(vaoID, MVP_loc, MVP, points_to_draw);


	float t = SDL_GetTicks() / 1000.0f;
	draw_cube(glm::translate(glm::vec3(0.5 + 0.7 * cos(t), 0, -0.4 + 0.7 * sin(t))));
	draw_rectangle(glm::mat4(1.0f));
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