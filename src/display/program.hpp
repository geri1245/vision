#ifndef DISPLAY_PROGRAM_HPP
#define DISPLAY_PROGRAM_HPP

#include <string>
#include <vector>
#include <fstream>
#include <iostream>

#include <GL/glew.h>

class Program
{
public:
	void generate_vao_vbo();
    
private:

};

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

#endif