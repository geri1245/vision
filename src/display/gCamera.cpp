#include <iostream>
#include "gCamera.h"
#include <glm/gtc/matrix_transform.hpp>
#include <math.h>

gCamera::gCamera() : eye(0.0f, 20.0f, 20.0f), at(0.0f), up(0.0f, 1.0f, 0.0f), speed(16.0f), goFw(0), goRight(0), slow(false)
{
	SetView( glm::vec3(0, 20, 20), glm::vec3(0,0,0), glm::vec3(0,1,0));

	dist = glm::length( at - eye );	

	SetProj(45.0f, 640/480.0f, 0.001f, 1000.0f);
}

gCamera::gCamera(glm::vec3 eye, glm::vec3 at, glm::vec3 up) : speed(16.0f), goFw(0), goRight(0), dist(10), slow(false)
{
	SetView(eye, at, up);
}

gCamera::~gCamera()
{
}

void gCamera::SetView(glm::vec3 eye, glm::vec3 at, glm::vec3 up)
{
	eye	= eye;
	at	= at;
	up	= up;

	fw  = glm::normalize( at - eye  );
	st = glm::normalize( glm::cross( fw, up ) );

	dist = glm::length( at - eye );	

	u = atan2f( fw.z, fw.x );
	v = acosf( fw.y );
}

void gCamera::SetProj(float angle, float aspect, float zn, float zf)
{
	matProj = glm::perspective( angle, aspect, zn, zf);
	matViewProj = matProj * viewMatrix;
}

glm::mat4 gCamera::GetViewMatrix()
{
	return viewMatrix;
}

void gCamera::Update(float deltaTime)
{
	eye += ( goFw * fw + goRight * st ) * speed * deltaTime;
	at  += ( goFw * fw + goRight * st ) * speed * deltaTime;

	viewMatrix = glm::lookAt( eye, at, up);
	matViewProj = matProj * viewMatrix;
}

void gCamera::UpdateUV(float du, float dv)
{
	u		+= du;
	v		 = glm::clamp<float>(v + dv, 0.1f, 3.1f);

	at = eye + dist*glm::vec3(	cosf(u)*sinf(v), 
										cosf(v), 
										sinf(u)*sinf(v) );

	fw = glm::normalize( at - eye );
	st = glm::normalize( glm::cross( fw, up ) );
}

void gCamera::SetSpeed(float val)
{
	speed = val;
}

void gCamera::Resize(int width, int height)
{
	matProj = glm::perspective(	45.0f, width / (float) height, 0.01f, 1000.0f );

	matViewProj = matProj * viewMatrix;
}

void gCamera::KeyboardDown(SDL_KeyboardEvent& key)
{
	switch ( key.keysym.sym )
	{
	case SDLK_LSHIFT:
	case SDLK_RSHIFT:
		if ( !slow )
		{
			slow = true;
			speed /= 4.0f;
		}
		break;
	case SDLK_w:
			goFw = 1;
		break;
	case SDLK_s:
			goFw = -1;
		break;
	case SDLK_a:
			goRight = -1;
		break;
	case SDLK_d:
			goRight = 1;
		break;
	}
}

void gCamera::KeyboardUp(SDL_KeyboardEvent& key)
{
	switch ( key.keysym.sym )
	{
	case SDLK_LSHIFT:
	case SDLK_RSHIFT:
		if ( slow )
		{
			slow = false;
			speed *= 4.0f;
		}
		break;
	case SDLK_w:
	case SDLK_s:
			goFw = 0;
		break;
	case SDLK_a:
	case SDLK_d:
			goRight = 0;
		break;
	}
}

void gCamera::MouseMove(SDL_MouseMotionEvent& mouse)
{
	if ( mouse.state & SDL_BUTTON_LMASK )
	{
		UpdateUV(mouse.xrel/100.0f, mouse.yrel/100.0f);
	}
}

void gCamera::LookAt(glm::vec3 at)
{
	SetView(eye, at, up);
}

