#pragma once

#include <SDL2/SDL.h>
#include <glm/glm.hpp>

class gCamera
{
public:
	gCamera();
	gCamera(glm::vec3 _eye, glm::vec3 _at, glm::vec3 _up);
	~gCamera();

	/// <summary>
	/// Gets the view matrix.
	/// </summary>
	/// <returns>The 4x4 view matrix</returns>
	glm::mat4 GetViewMatrix();

	void Update(float _deltaTime);

	void SetView(glm::vec3 _eye, glm::vec3 _at, glm::vec3 _up);
	void SetProj(float _angle, float _aspect, float _zn, float _zf); 
	void LookAt(glm::vec3 _at);

	void SetSpeed(float _val);
	glm::vec3 GetEye()
	{
		return m_eye;
	}

	glm::vec3 GetAt()
	{
		return m_at;
	}

	glm::vec3 GetUp()
	{
		return m_up;
	}

	glm::mat4 GetProj()
	{
		return m_matProj;
	}

	glm::mat4 GetViewProj()
	{
		return m_matViewProj;
	}

	void Resize(int _w, int _h);

	void KeyboardDown(SDL_KeyboardEvent& key);
	void KeyboardUp(SDL_KeyboardEvent& key);
	void MouseMove(SDL_MouseMotionEvent& mouse);

private:
	void UpdateUV(float du, float dv);

	glm::vec3	m_eye;
	glm::vec3	m_at;
	glm::vec3	m_up;

	float	m_speed;
	float	m_goFw;
	float	m_goRight;
	float	m_dist;

	bool	m_slow;

	glm::mat4	m_viewMatrix;
	glm::mat4	m_matViewProj;

	float	m_u;
	float	m_v;

	glm::vec3	m_fw;
	glm::vec3	m_st;

	glm::mat4	m_matProj;

};

