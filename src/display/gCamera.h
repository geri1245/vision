#ifndef DISPLAY_GCAMERA_H
#define DISPLAY_GCAMERA_H

#include <SDL2/SDL.h>
#include <glm/glm.hpp>

class gCamera
{
public:
	gCamera();
	gCamera(glm::vec3 eye, glm::vec3 at, glm::vec3 up);
	~gCamera();

	glm::mat4 GetViewMatrix();

	void Update(float deltaTime);

	void SetView(glm::vec3 eye, glm::vec3 at, glm::vec3 up);
	void SetProj(float angle, float aspect, float zn, float zf); 
	void LookAt(glm::vec3 at);

	void SetSpeed(float val);
	glm::vec3 GetEye()      { return eye;         }
	glm::vec3 GetAt()       { return at;          }
	glm::vec3 GetUp()       { return up;          }
	glm::mat4 GetProj()     { return matProj;     }
	glm::mat4 GetViewProj() { return matViewProj; }

	void Resize(int width, int height);

	void KeyboardDown(SDL_KeyboardEvent& key);
	void KeyboardUp(SDL_KeyboardEvent& key);
	void MouseMove(SDL_MouseMotionEvent& mouse);

private:
	void UpdateUV(float du, float dv);

	glm::vec3	eye;
	glm::vec3	at;
	glm::vec3	up;

	float	speed;
	float	goFw;
	float	goRight;
	float	dist;

	bool	slow;

	glm::mat4	viewMatrix;
	glm::mat4	matViewProj;

	float	u;
	float	v;

	glm::vec3	fw;
	glm::vec3	st;

	glm::mat4	matProj;

};

#endif