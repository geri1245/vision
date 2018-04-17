#ifndef DISPLAY_PROGRAM_HPP
#define DISPLAY_PROGRAM_HPP

#include <string>
#include <vector>
#include <fstream>
#include <iostream>

#include <GL/glew.h>
#include <glm/glm.hpp>


class Program
{
public:

	Program();
	~Program();

	template <typename T>
	void generate_vao_vbo(
		GLuint &vaoID, GLuint &vboID, int size, 
		const T *data, GLuint mode = GL_DYNAMIC_DRAW);

	void create_program_with_shaders(
		const std::string &vert_path, 
		const std::string &frag_path);

	void get_uniform(GLuint &uniform_loc, const std::string &name);

	template <typename T>
	void update_vbo(GLuint vboID, int size, const T *data);

	void draw_points(
		GLuint vaoID, GLuint MVP_loc, 
		const glm::mat4 &MVP, int num_points);

	void clean(GLuint vaoID, GLuint vboID);
	GLuint program_id();

private:

	GLuint programID;
};


//Local function to read the source code of a shader and compile it

namespace{

	inline GLuint load_shader(GLenum shader_type, const std::string &filename)
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


inline Program::Program() 
{
	programID = glCreateProgram();
}

inline Program::~Program() 
{
	glDeleteProgram(programID);
}

template <typename T>
void Program::generate_vao_vbo(GLuint &vaoID, GLuint &vboID, int size, const T *data, GLuint mode)
{
	glGenVertexArrays(1, &vaoID);
	glBindVertexArray(vaoID);
	
	glGenBuffers(1, &vboID); 

	glBindBuffer(GL_ARRAY_BUFFER, vboID);
	glBufferData(GL_ARRAY_BUFFER, size, data, mode);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer( (GLuint)0, 3, GL_FLOAT, GL_FALSE, sizeof(T), 0 ); 

	glEnableVertexAttribArray(1);
	glVertexAttribPointer(
		(GLuint)1,
		3, 
		GL_FLOAT,
		GL_FALSE,
		sizeof(T),
		(void*)(sizeof(glm::vec3)) );

	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}


inline void Program::create_program_with_shaders(
	const std::string &vert_path, 
	const std::string &frag_path)
{
	GLuint vs_ID = load_shader(GL_VERTEX_SHADER,   vert_path);
	GLuint fs_ID = load_shader(GL_FRAGMENT_SHADER, frag_path);

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
}


inline void Program::get_uniform(GLuint &uniform_loc, const std::string &name)
{
	uniform_loc = glGetUniformLocation(programID, name.c_str());
}


template <typename T>
inline void Program::update_vbo(GLuint vboID, int size, const T *data)
{
	glBindBuffer(GL_ARRAY_BUFFER, vboID);
	glBufferData(GL_ARRAY_BUFFER, size, data, GL_DYNAMIC_DRAW);
}


inline void Program::draw_points(GLuint vaoID, GLuint MVP_loc, const glm::mat4 &MVP, int num_points)
{
	glUseProgram( programID );

	glUniformMatrix4fv( MVP_loc, 1, GL_FALSE, &(MVP[0][0]) );

	glBindVertexArray(vaoID);

	glDrawArrays(GL_POINTS, 0, num_points);

	glBindVertexArray(0);

	glUseProgram( 0 );
}


inline void Program::clean(GLuint vaoID, GLuint vboID)
{
	glDeleteBuffers(1, &vboID);
	glDeleteVertexArrays(1, &vaoID);
}

inline GLuint Program::program_id()
{
	return programID;
}

#endif