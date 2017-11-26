#define GL_FUNCTION(type, name, prototype) \
    type(APIENTRY * name) prototype = \
        (type(APIENTRY *) prototype)(get_gl_function("gl" #name))

GL_FUNCTION(GLenum, GetError, (void));
GL_FUNCTION(void, Enable, (GLenum));
GL_FUNCTION(void, Clear, (GLbitfield));
GL_FUNCTION(void, Disable, (GLenum));
GL_FUNCTION(void, FrontFace, (GLenum));
GL_FUNCTION(void, CullFace, (GLenum));
GL_FUNCTION(void, ClearColor, (GLclampf, GLclampf, GLclampf, GLclampf));
GL_FUNCTION(GLint, GetUniformLocation, (GLuint, const GLchar*));
GL_FUNCTION(void, Viewport, (GLint, GLint, GLsizei, GLsizei));

GL_FUNCTION(void, GenVertexArrays, (GLsizei, GLuint*));
GL_FUNCTION(void, GenBuffers, (GLsizei, GLuint*));
GL_FUNCTION(void, BindVertexArray, (GLuint));
GL_FUNCTION(void, BindBuffer, (GLenum, GLuint));
GL_FUNCTION(void, BufferData, (GLenum, GLsizeiptr, const GLvoid*, GLenum));
GL_FUNCTION(void, VertexAttribPointer, (GLuint, GLint, GLenum, GLboolean, GLsizei, const GLvoid*));
GL_FUNCTION(void, VertexAttribDivisor, (GLuint, GLuint));
GL_FUNCTION(void, EnableVertexAttribArray, (GLuint));
GL_FUNCTION(void, DisableVertexAttribArray, (GLuint));
GL_FUNCTION(void, Uniform1i, (GLint, GLint));
GL_FUNCTION(void, Uniform1f, (GLint, GLfloat));
GL_FUNCTION(void, Uniform3fv, (GLint, GLsizei, GLfloat const*));
GL_FUNCTION(void, Uniform4fv, (GLint, GLsizei, GLfloat const*));
GL_FUNCTION(void, UniformMatrix4fv, (GLint, GLsizei, GLboolean, const GLfloat*));
GL_FUNCTION(void, DrawElements, (GLenum, GLsizei, GLenum, const GLvoid*));
GL_FUNCTION(void, DrawElementsInstanced, (GLenum, GLsizei, GLenum, const GLvoid*, GLsizei));

GL_FUNCTION(GLuint, CreateProgram, (void));
GL_FUNCTION(GLuint, CreateShader, (GLenum));
GL_FUNCTION(void, ShaderSource, (GLuint, GLsizei, const GLchar**, const GLint*));
GL_FUNCTION(void, CompileShader, (GLuint));
GL_FUNCTION(void, LinkProgram, (GLuint));
GL_FUNCTION(void, AttachShader, (GLuint, GLuint));
GL_FUNCTION(void, GetShaderiv, (GLuint, GLenum, GLint*));
GL_FUNCTION(void, GetProgramiv, (GLuint, GLenum, GLint*));
GL_FUNCTION(void, GetShaderInfoLog, (GLuint, GLsizei, GLsizei*, GLchar*));
GL_FUNCTION(void, GetProgramInfoLog, (GLuint, GLsizei, GLsizei*, GLchar*));
GL_FUNCTION(void, UseProgram, (GLuint));

GL_FUNCTION(void, GenTextures, (GLsizei, GLuint*));
GL_FUNCTION(void, BindTexture, (GLenum, GLuint));
GL_FUNCTION(void, TexImage2D, (GLenum, GLint, GLint, GLsizei, GLsizei, GLint, GLenum, GLenum, const GLvoid*));
GL_FUNCTION(void, ActiveTexture, (GLenum));
GL_FUNCTION(void, GenerateMipmap, (GLenum));
GL_FUNCTION(void, TexParameteri, (GLenum, GLenum, GLint));
GL_FUNCTION(void, BlendFunc, (GLenum, GLenum));

