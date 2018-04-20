#version 130

in vec3 vs_out_col;
in vec3 vs_out_pos;

uniform float alpha;

out vec4 fs_out_col;

void main()
{
	fs_out_col = vec4(vs_out_col, alpha);
}