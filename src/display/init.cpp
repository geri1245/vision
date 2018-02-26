#include <GL/glew.h>
#include <GL/glut.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glext.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>

#include <iostream>
#include <sstream>
#include <vector>

#include "displayer.h"
#include "init.h"

int init_display()
{
	if ( SDL_Init( SDL_INIT_VIDEO ) == -1 )
	{
		std::cerr << "[SDL_Init] Error while initializing SDL: " 
                  << SDL_GetError() 
                  << std::endl;
		return 1;
	}
			
    //Set OpenGL flags

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);

    SDL_GL_SetAttribute(SDL_GL_BUFFER_SIZE,         32);
    SDL_GL_SetAttribute(SDL_GL_RED_SIZE,            8);
    SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE,          8);
    SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE,           8);
    SDL_GL_SetAttribute(SDL_GL_ALPHA_SIZE,          8);

	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER,		1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE,          24);


    SDL_DisplayMode DisplayMode;
    SDL_GetCurrentDisplayMode(0, &DisplayMode);
    auto width = DisplayMode.w;
    auto height = DisplayMode.h;

	SDL_Window *win = nullptr;
    win = SDL_CreateWindow( "3D Lidar point visualization",
							0,
							0,
							width,
							height,
							SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);


    if (win == nullptr)
	{
		std::cout << "[Creating window] Error while creating window: " 
                  << SDL_GetError() 
                  << std::endl;
        return 1;
    }

	SDL_GLContext context = SDL_GL_CreateContext(win);
    if (context == nullptr)
	{
		std::cout << "[Creating OGL context] Error while creating OpenGL context: "
                  << SDL_GetError() 
                  << std::endl;
        return 1;
    }	

	SDL_GL_SetSwapInterval(1);

    //GLEW

	GLenum error = glewInit();
	if ( error != GLEW_OK )
	{
		std::cout << "[GLEW Init] Error while initializing GLEW!" << std::endl;
		return 1;
	}

	int glVersion[2] = {-1, -1}; 
	glGetIntegerv(GL_MAJOR_VERSION, &glVersion[0]); 
	glGetIntegerv(GL_MINOR_VERSION, &glVersion[1]);

	if ( glVersion[0] == -1 && glVersion[1] == -1 )
	{
		SDL_GL_DeleteContext(context);
		SDL_DestroyWindow( win );

		std::cout << "[Creating OGL context]" 
                  << "Error while creating OpenGL context." 
                  << std::endl;
		return 1;
	}

	std::stringstream window_title;
	window_title << "Point visualization with OpenGL " << glVersion[0] << "." << glVersion[1];
	SDL_SetWindowTitle(win, window_title.str().c_str());


    //Main loop

	bool quit = false;
	SDL_Event ev;
	
	Displayer displayer;
	if (!displayer.init())
	{
		SDL_GL_DeleteContext(context);
		SDL_DestroyWindow(win);
		std::cout << "[Application Init] " 
        << "Error while initializing the application" << std::endl;
		return 1;
	}

	while (!quit)
	{
		//Propagating events to the application
		while ( SDL_PollEvent(&ev) )
		{
			switch(ev.type)
            {
            case SDL_QUIT:
				quit = true;
				break;
            case SDL_KEYDOWN:
				if ( ev.key.keysym.sym == SDLK_ESCAPE )
					quit = true;
				break;
            }
            displayer.handle_event(ev);
        }

		displayer.update();
		displayer.render();

		SDL_GL_SwapWindow(win);
	}

    displayer.clean();

	SDL_GL_DeleteContext(context);
	SDL_DestroyWindow( win );

	return 0;
}