/*    SDL2, OpenGL 3.3, GLM client for Jelly McJelloFace - David Akeley 2017
 *    
 *  The game simulates a 1x1x1 jello cube with beads embedded inside of  it.
 *  The  beads  are  unartistically  arranged in the shape of a face and the
 *  cube  is  drawn  against  a  skybox  background,  with  that  background
 *  reflecting  and  refracting  off the cube. The player interacts with the
 *  cube by delivering jolts and using a laser to melt down the cube.
 *  
 *  This file implements the client of the Jelly McJelloFace  program.  It's
 *  split  up  into six logical portions: part 1 (implemented in beadface.h)
 *  implements the hierarchical code for arranging beads in the shape  of  a
 *  face.  Part  2  implements  uninteresting bookkeeping for loading OpenGL
 *  functions, setting up SDL2, and various  utility  functions  for  things
 *  like  compiling  shaders,  loading  cubemap textures, and panic (part of
 *  this  is  split  into  opengl-functions.inc).  Parts  3  -  5  implement
 *  functions for drawing the objects in the program: the beads, the skybox,
 *  and the distorting, reflective and refractive jello  cube.  Each  object
 *  has  its  data encapsulated in a single OpenGL vertex array object (vao)
 *  that is initialized only once, the first time the object is drawn.  Part
 *  6  implements  the  controls  and main loop, which configures OpenGL and
 *  coordinates calling the object drawing functions.
 */

#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <string>

#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

#include "SDL2/SDL.h"
#include "SDL2/SDL_opengl.h"

#include "vec3.h"
#include "jellymjf.h"
#include "beadface.h"

static int screen_x = 1280;
static int screen_y = 960;
static const float fovy_radians = 1.0f;
static const float near_plane = 0.01f;
static const float far_plane = 40.0f;

static SDL_GLContext gl_context = nullptr;
static SDL_Window* window = nullptr;
static std::string argv0;
static bool cubemap_loaded = false;
static GLuint cubemap_texture_id = 0;

static glm::mat4 view;
static glm::mat4 projection;
static glm::vec3 eye;

static glm::vec4 laser_direction(1, 0, 0, 0);
static const float laser_heat_per_second = 60.0f;
static const float laser_beam_radius = 0.4f;
static bool laser_on = false;
static bool beads_drawn = true;
static bool bead_debug = false;

/*  Part 2: Boring utility functions.
 */

static void panic(const char* message, const char* reason) {
    fprintf(stderr, "JellyMcJelloFace: %s %s\n", message, reason);
    fflush(stderr);
    fflush(stdout);
    SDL_ShowSimpleMessageBox(
        SDL_MESSAGEBOX_ERROR, message, reason, nullptr
    );
    exit(1);
    abort();
}

static void show_controls(void) {
    SDL_ShowSimpleMessageBox(
        SDL_MESSAGEBOX_INFORMATION,
        "Jelly McJelloFace",
        
        "'/', '?', or P to bring up this menu.\n"
        "Orbit view mode: QEAD/UOJL to orbit, WS/IK to zoom.\n"
        "Free view mode: WS/IK forward movement, AD/JL sideways movement\n"
        "   QE/UO vertical movement, hold space to look around.\n"
        "Two-finger scroll can also be used for orbit/look around.\n"
        "X / ',' to change view modes.\n"
        "Click and drag to use the laser.\n"
        "Tab/enter to deliver a jolt to the cube.\n"
        "F/H to toggle bead visibility.\n"
        "Shift + F/H to view debug beads.\n"
        "Shift+R to reset.\n",
        nullptr
    );
}

// OpenGL functions will be accessed through function pointers stored in a
// struct named gl, instead of using the typical C gl* forms (e.g. glEnable
// becomes gl.Enable). SDL2 is tasked with loading these function pointers.
// I do this because I don't feel like dealing with cross-platform OpenGL
// context setup and loading (especially not on Win Doze).
static void* get_gl_function(const char* name) {
    static bool initialized = false;
    
    if (!initialized) {
        window = SDL_CreateWindow(
            "Jelly McJelloFace",
            SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
            screen_x, screen_y,
            SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE
        );
        if (window == nullptr) {
            panic("Could not initialize window", SDL_GetError());
        }
        // OpenGL 3.3 needed for delicious instanced rendering.
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
        gl_context = SDL_GL_CreateContext(window);
        
        if (gl_context == nullptr) {
            panic("Could not initialize OpenGL 3.3", SDL_GetError());
        }
        initialized = true;
    }
    
    void* result = SDL_GL_GetProcAddress(name);
    if (result == nullptr) panic(name, "Missing OpenGL function");
    return result;
}

struct OpenGL_Functions {

#include "opengl-functions.inc"

};

typedef OpenGL_Functions const& GL;

#define PANIC_IF_GL_ERROR(gl) do { \
    if (auto PANIC_error = gl.GetError()) { \
        char PANIC_msg[30]; \
        sprintf(PANIC_msg, "line %i: code %i", __LINE__, (int)PANIC_error); \
        panic("OpenGL error", PANIC_msg); \
    } \
} while (0)

static GLuint make_program(GL gl, const char* vs_code, const char* fs_code) {
    static GLchar log[1024];
    GLuint program_id = gl.CreateProgram();
    GLuint vs_id = gl.CreateShader(GL_VERTEX_SHADER);
    GLuint fs_id = gl.CreateShader(GL_FRAGMENT_SHADER);
    
    const GLchar* string_array[1];
    string_array[0] = (GLchar*)vs_code;
    gl.ShaderSource(vs_id, 1, string_array, nullptr);
    string_array[0] = (GLchar*)fs_code;
    gl.ShaderSource(fs_id, 1, string_array, nullptr);
    
    gl.CompileShader(vs_id);
    gl.CompileShader(fs_id);
    
    GLint okay = 0;
    GLsizei length = 0;
    const GLuint shader_id_array[2] = { vs_id, fs_id };
    for (auto id : shader_id_array) {
        gl.GetShaderiv(id, GL_COMPILE_STATUS, &okay);
        if (okay) {
            gl.AttachShader(program_id, id);
        } else {
            gl.GetShaderInfoLog(id, sizeof log, &length, log);
            fprintf(stderr, "%s\n", id == vs_id ? vs_code : fs_code);
            panic("Shader compilation error", log);
        }
    }
    
    gl.LinkProgram(program_id);
    gl.GetProgramiv(program_id, GL_LINK_STATUS, &okay);
    if (!okay) {
        gl.GetProgramInfoLog(program_id, sizeof log, &length, log);
        panic("Shader link error", log);
    }
    
    PANIC_IF_GL_ERROR(gl);
    return program_id;
}

static void load_cubemap_face(GL gl, GLenum face, const char* filename) {
    std::string full_filename = argv0 + "Tex/" + filename;
    SDL_Surface* surface = SDL_LoadBMP(full_filename.c_str());
    if (surface == nullptr) {
        panic(SDL_GetError(), full_filename.c_str()); 
    }
    if (surface->w != 512 || surface->h != 512) {
        panic("Expected 512x512 texture", full_filename.c_str());
    }
    if (surface->format->format != SDL_PIXELFORMAT_BGR24) {
        fprintf(stderr, "%i\n", (int)surface->format->format);
        panic("Expected 24-bit BGR bitmap", full_filename.c_str());
    }
    
    gl.TexImage2D(face, 0, GL_RGB, 512, 512, 0,
                  GL_BGR, GL_UNSIGNED_BYTE, surface->pixels);
    
    SDL_FreeSurface(surface);
}

static GLuint load_cubemap(GL gl) {
    GLuint id = 0;
    gl.GenTextures(1, &id);
    gl.BindTexture(GL_TEXTURE_CUBE_MAP, id);
    
    gl.TexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    gl.TexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    gl.TexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    gl.TexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    gl.TexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    gl.TexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_LOD, 0);
    gl.TexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAX_LOD, 8);
    gl.TexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_BASE_LEVEL, 0); 
    gl.TexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAX_LEVEL, 8); 
    
    load_cubemap_face(gl, GL_TEXTURE_CUBE_MAP_NEGATIVE_X, "left.bmp");
    load_cubemap_face(gl, GL_TEXTURE_CUBE_MAP_POSITIVE_X, "right.bmp");
    load_cubemap_face(gl, GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, "bottom.bmp");
    load_cubemap_face(gl, GL_TEXTURE_CUBE_MAP_POSITIVE_Y, "top.bmp");
    load_cubemap_face(gl, GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, "back.bmp");
    load_cubemap_face(gl, GL_TEXTURE_CUBE_MAP_POSITIVE_Z, "front.bmp");
    
    gl.GenerateMipmap(GL_TEXTURE_CUBE_MAP);
    
    PANIC_IF_GL_ERROR(gl);
    gl.BindTexture(GL_TEXTURE_CUBE_MAP, 0);
    
    return id;
}


/*  Part 3: Code for drawing beads.
 *  
 *  Each bead is drawn as a regular icosahedron. It's the lowest-poly  shape
 *  I can think of for drawing roughly spherical objects. To reduce overhead
 *  for drawing thousands of beads,  I'm  using  instanced  rendering.  Each
 *  vertex  of  a  single  icosahedron has four attributes: its position and
 *  normal in the coordinate system  of  a  single  icosahedron  (center  at
 *  origin),  and  the  color  and  position  in space of the bead the drawn
 *  icosahedron represents. The former 2 attributes  come  from  icosahedron
 *  vertex  data  defined  in this file (bead vertices) filled in the vertex
 *  buffer with id [vertex_buffer_id]; the latter  2  attributes  come  from
 *  data  exported  from  the  physics  backend  of the program that will be
 *  stored in the vertex buffer [instanced_buffer_id], and will  have  their
 *  attribute  divisor  set  to  1  so  that the color and position in space
 *  changes once per icosahedron, not once per icosahedron vertex.
 */

static const GLuint vertex_position_index = 0;
static const GLuint vertex_normal_index = 1;
static const GLuint instance_position_index = 1;
static const GLuint instance_color_index = 2;

static const char bead_vs_source[] =
"#version 330\n"
"precision mediump float;\n"
"layout(location=0) in vec3 vertex_position;\n"
"layout(location=1) in vec3 instance_position;\n"
"layout(location=2) in vec3 instance_color;\n"
"out vec3 material_color;\n"
"out vec4 varying_normal;\n"
"uniform mat4 view_matrix;\n"
"uniform mat4 proj_matrix;\n"
"void main() {\n"
    "mat4 VP = proj_matrix * view_matrix;\n"
    "gl_Position = VP * vec4(vertex_position + instance_position, 1.0);\n"
    "material_color = instance_color;\n"
    "varying_normal = view_matrix * vec4(vertex_position, 0.0);\n"
    // The vertex normal is the same as its position for spherical objects.
"}\n";

static const char bead_fs_source[] =
"#version 330\n"
"precision mediump float;\n"
"in vec3 material_color;\n"
"in vec4 varying_normal;\n"
"out vec4 pixel_color;\n"
"void main() {\n"
    "float z = normalize(varying_normal.xyz).z;\n"
    "pixel_color = vec4(material_color * sqrt(z*.8 + .2), 1.0);\n"
"}\n";

// Each bead will be a sphere approximated by a regular icosahedron.
static const int bead_vertex_count = 12;
static const int bead_element_count = 60;
static const float bead_scale = 5e-3f;
static const float phi = 1.618034f;
static const float bead_vertices[3 * bead_vertex_count] = {
     phi*bead_scale,  bead_scale, 0,
    -phi*bead_scale,  bead_scale, 0,
    -phi*bead_scale, -bead_scale, 0,
     phi*bead_scale, -bead_scale, 0,
     
     bead_scale, 0,  phi*bead_scale,
    -bead_scale, 0,  phi*bead_scale,
    -bead_scale, 0, -phi*bead_scale,
     bead_scale, 0, -phi*bead_scale,
     
     0,  phi*bead_scale,  bead_scale,
     0, -phi*bead_scale,  bead_scale,
     0, -phi*bead_scale, -bead_scale,
     0,  phi*bead_scale, -bead_scale,
};

static const GLushort bead_elements[bead_element_count] = {
    5, 4, 8,
    5, 8, 1,
    5, 1, 2,
    5, 2, 9,
    5, 9, 4,
    
    7, 6, 11,
    7, 11, 0,
    7, 0, 3,
    7, 3, 10,
    7, 10, 6,
    
    2, 1, 6,
    6, 1, 11,
    1, 8, 11,
    11, 8, 0,
    8, 4, 0,
    0, 4, 3,
    4, 9, 3,
    3, 9, 10,
    9, 2, 10,
    10, 2, 6,
};

static void draw_beads(GL gl) {
    static GLuint vao = 0;
    static GLuint program_id;
    static GLuint vertex_buffer_id;
    static GLuint element_buffer_id;
    static GLuint instance_buffer_id;
    static GLint view_matrix_id;
    static GLint proj_matrix_id;
    static int bead_vertex_stride = get_bead_vertex_stride();
    
    if (!beads_drawn) return;
    
    if (vao == 0) {
        program_id = make_program(gl, bead_vs_source, bead_fs_source);
        view_matrix_id = gl.GetUniformLocation(program_id, "view_matrix");
        proj_matrix_id = gl.GetUniformLocation(program_id, "proj_matrix");
        gl.GenVertexArrays(1, &vao);
        gl.BindVertexArray(vao);
        
        gl.GenBuffers(1, &vertex_buffer_id);
        gl.GenBuffers(1, &element_buffer_id);
        gl.GenBuffers(1, &instance_buffer_id);
        
        gl.BindBuffer(GL_ARRAY_BUFFER, vertex_buffer_id);
        gl.BufferData(
            GL_ARRAY_BUFFER, sizeof bead_vertices, bead_vertices, GL_STATIC_DRAW
        );
        gl.BindBuffer(GL_ELEMENT_ARRAY_BUFFER, element_buffer_id);
        gl.BufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof bead_elements,
            bead_elements, GL_STATIC_DRAW
        );
        gl.VertexAttribPointer(
            vertex_position_index,
            3,
            GL_FLOAT,
            false,
            3 * sizeof(float),
            (void*)0
        );
        
        gl.BindBuffer(GL_ARRAY_BUFFER, instance_buffer_id);
        gl.VertexAttribPointer(
            instance_position_index,
            3,
            GL_FLOAT,
            false,
            bead_vertex_stride,
            (void*)0
        );
        gl.VertexAttribPointer(
            instance_color_index,
            3,
            GL_FLOAT,
            false,
            bead_vertex_stride,
            (void*)(intptr_t)get_bead_color_offset()
        );
        gl.VertexAttribDivisor(instance_position_index, 1);
        gl.VertexAttribDivisor(instance_color_index, 1);
        
        gl.EnableVertexAttribArray(vertex_position_index);
        gl.EnableVertexAttribArray(instance_position_index);
        gl.EnableVertexAttribArray(instance_color_index);
        
        PANIC_IF_GL_ERROR(gl);
    }
    
    gl.UseProgram(program_id);
    
    gl.UniformMatrix4fv(view_matrix_id, 1, 0, &view[0][0]);
    gl.UniformMatrix4fv(proj_matrix_id, 1, 0, &projection[0][0]);
    
    gl.BindVertexArray(vao);
    gl.BindBuffer(GL_ARRAY_BUFFER, instance_buffer_id);
    
    int bead_count = bead_debug ? get_debug_bead_count() : get_bead_count();
    BeadVertex const* bead_ptr = 
        bead_debug ? update_debug_bead_vertices() : update_bead_vertices();
    
    gl.BufferData(
        GL_ARRAY_BUFFER,
        bead_vertex_stride * bead_count,
        bead_ptr,
        GL_DYNAMIC_DRAW
    );
    gl.DrawElementsInstanced(
        GL_TRIANGLES,
        bead_element_count,
        GL_UNSIGNED_SHORT,
        (void*)0,
        bead_count
    );
    
    gl.BindVertexArray(0);
    PANIC_IF_GL_ERROR(gl);
}

/*  Part 4: Drawing a skybox.
 *  
 *  Not really that sophisticated code, just verbose. All we have to  do  is
 *  draw  a big cube (40x40x40) around the camera and map the skybox cubemap
 *  to it. To keep the skybox around the camera we make the  w-component  of
 *  "position"  0 instead of 1 when we multiply it by the view matrix in the
 *  vertex shader.
 */
static const char skybox_vs_source[] =
"#version 330\n"
"layout(location=0) in vec3 position;\n"
"out vec3 texture_coordinate;\n"
"uniform mat4 view_matrix;\n"
"uniform mat4 proj_matrix;\n"
"void main() {\n"
    "vec4 v = view_matrix * vec4(20*position, 0.0);\n"
    "gl_Position = proj_matrix * vec4(v.xyz, 1);\n"
    "texture_coordinate = position;\n"
"}\n";

static const char skybox_fs_source[] =
"#version 330\n"
"in vec3 texture_coordinate;\n"
"out vec4 color;\n"
"uniform samplerCube cubemap;\n"
"void main() {\n"
    "vec4 c = texture(cubemap, texture_coordinate);\n"
    "c.a = 1.0;\n"
    "color = c;\n"
"}\n";

static const float skybox_vertices[24] = {
    -1, 1, 1,
    -1, -1, 1,
    1, -1, 1,
    1, 1, 1,
    -1, 1, -1,
    -1, -1, -1,
    1, -1, -1,
    1, 1, -1,
};

static const GLushort skybox_elements[36] = {
    7, 4, 5, 7, 5, 6,
    1, 0, 3, 1, 3, 2,
    5, 1, 2, 5, 2, 6,
    4, 7, 3, 4, 3, 0,
    0, 1, 5, 0, 5, 4,
    2, 3, 7, 2, 7, 6
};

static void draw_skybox(GL gl) {
    if (!cubemap_loaded) {
        cubemap_texture_id = load_cubemap(gl);
        cubemap_loaded = true;
    }
    
    static GLuint vao = 0;
    static GLuint program_id;
    static GLuint vertex_buffer_id;
    static GLuint element_buffer_id;
    static GLint view_matrix_id;
    static GLint proj_matrix_id;
    static GLint cubemap_uniform_id;
    
    if (vao == 0) {
        program_id = make_program(gl, skybox_vs_source, skybox_fs_source);
        view_matrix_id = gl.GetUniformLocation(program_id, "view_matrix");
        proj_matrix_id = gl.GetUniformLocation(program_id, "proj_matrix");
        cubemap_uniform_id = gl.GetUniformLocation(program_id, "cubemap");
        
        gl.GenVertexArrays(1, &vao);
        gl.BindVertexArray(vao);
        
        gl.GenBuffers(1, &vertex_buffer_id);
        gl.GenBuffers(1, &element_buffer_id);
        
        gl.BindBuffer(GL_ELEMENT_ARRAY_BUFFER, element_buffer_id);
        gl.BufferData(
            GL_ELEMENT_ARRAY_BUFFER, sizeof skybox_elements,
            skybox_elements, GL_STATIC_DRAW
        );
        
        gl.BindBuffer(GL_ARRAY_BUFFER, vertex_buffer_id);
        gl.BufferData(
            GL_ARRAY_BUFFER, sizeof skybox_vertices,
            skybox_vertices, GL_STATIC_DRAW
        ); 
        gl.VertexAttribPointer(
            0,
            3,
            GL_FLOAT,
            false,
            sizeof(float) * 3,
            (void*)0
        );
        gl.EnableVertexAttribArray(0);
        PANIC_IF_GL_ERROR(gl);
    }
    
    gl.UseProgram(program_id);
    
    gl.ActiveTexture(GL_TEXTURE0);
    gl.BindTexture(GL_TEXTURE_CUBE_MAP, cubemap_texture_id);
    gl.Uniform1i(cubemap_uniform_id, 0);
    
    gl.UniformMatrix4fv(view_matrix_id, 1, 0, &view[0][0]);
    gl.UniformMatrix4fv(proj_matrix_id, 1, 0, &projection[0][0]);
    
    gl.BindVertexArray(vao);
    gl.DrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_SHORT, (void*)0);
    gl.BindVertexArray(0);
    gl.BindTexture(GL_TEXTURE_CUBE_MAP, 0);
    
    PANIC_IF_GL_ERROR(gl);
}

/*  Part 5: Drawing the jello cube, which reflects and refracts  the  skybox
 *  texture.  This  is  called twice per frame (at time of writing); once to
 *  draw the interior faces (back face) of the cube and  once  to  draw  the
 *  exterior  front  faces.  The glCullFace state determines this function's
 *  behavior. It may have been more efficient to write 2 separate  functions
 *  and shaders but I didn't.
 *  
 *  There's a lot going on in this code. The basic idea  is  that  we  first
 *  depend  on  the  physics  backend  to  give us data each frame about the
 *  triangles on the 6 surfaces of the jello cube: one position  and  normal
 *  for  each  node  on the surface, with that data stored in an interleaved
 *  array. Then, in the shader, for each vertex we use that normal  and  the
 *  eye  position  to  calculate  two  vectors:  a  reflection  vector and a
 *  refraction vector, which are later used as texture  coordinates  in  the
 *  fragment  shader. To help make edges and geometry pop out a bit more, we
 *  also calculate an "ambient" light component, whose brightness  increases
 *  the closer the normal is to pointing directly at the camera.
 *  
 *  In the fragment shader, we add up the light from  the  texture  (fetched
 *  using  the  vectors from earlier), the ambient light, and light from the
 *  laser to get the  fragment  color.  This  laser  color  depends  on  how
 *  directly  the  laser  is  shining  on this particular fragment. If we're
 *  rendering a backface, the laser color  is  ignored  and  we  sample  the
 *  texture   using   the   refraction   vector  and  an  intentionally  low
 *  level-of-detail to make the refraction look blurry. As  there's  nothing
 *  but sky behind the cube, the fragment is opaque (alpha=1); it only looks
 *  transparent from having sampled from the same skybox texture.  If  we're
 *  rendering  a  front  face,  we  include the laser color and the texel(s)
 *  sampled using the reflection vector, and make the  fragment  transparent
 *  so that the beads within the cube are still shown.
 */

static const char cube_vs_source[] =
"#version 330\n"
"layout(location=0) in vec3 vertex_position;\n"
"layout(location=1) in vec3 vertex_normal;\n"
"out vec3 reflected_vector;\n"
"out vec3 refracted_vector;\n"
"out vec3 front_ambient;\n"
"out vec3 back_ambient;\n"
"out vec3 varying_position;\n"
"uniform vec3 eye;\n"
"uniform mat4 view_matrix;\n"
"uniform mat4 proj_matrix;\n"
"void main() {\n"
    "vec3 n = normalize(vertex_normal);\n"
    "mat4 VP = proj_matrix * view_matrix;\n"
    "gl_Position = VP * vec4(vertex_position, 1);\n"
    "vec3 eye_vertex = vertex_position - eye;\n"
    // This refracted vector is fake af because I ran out of physics mana at 1 am.
    "refracted_vector = eye_vertex - 0.175*dot(eye_vertex, n) * n;\n"
    "reflected_vector = eye_vertex - 2*dot(eye_vertex, n) * n;\n"
    "float z = abs(normalize(view_matrix * vec4(vertex_normal, 0))).z;\n"
    "float ambient = sqrt(z * 0.6 + 0.4);\n"
    "back_ambient = ambient * vec3(0.06, 0.06, 0.06);\n"
    "front_ambient = ambient * vec3(0.15, 0.05, 0.10);\n"
    "varying_position = vertex_position;\n"
"}\n";

static const char cube_fs_source[] =
"#version 330\n"
"in vec3 reflected_vector;\n"
"in vec3 refracted_vector;\n"
"in vec3 front_ambient;\n"
"in vec3 back_ambient;\n"
"in vec3 varying_position;\n"
"out vec4 color;\n"
"uniform samplerCube cubemap;\n"
"uniform vec3 eye;\n"
"uniform vec3 laser_direction;\n"
"uniform vec4 laser_color;\n"
"uniform float laser_radius;\n"
"void main() {\n"
    "vec4 front_texel = texture(cubemap, reflected_vector);\n"
    "vec4 back_texel = textureLod(cubemap, refracted_vector, 2.5);\n"
    "vec4 front_color = vec4(front_texel.rgb + front_ambient, 0.4);\n"
    "vec4 back_color = vec4(back_texel.rgb * 0.5 + back_ambient, 1.0);\n"
    
    "vec3 v = normalize(laser_direction);\n"
    "vec3 displacement = varying_position - eye;\n"
    "vec3 beam_projection = dot(displacement, v) * v;\n"
    "vec3 perpendicular = displacement - beam_projection;\n"
    "float r = laser_radius, ss = dot(perpendicular, perpendicular);\n"
    "float brightness = clamp(1.5*r*r - ss, 0, 1);\n"
    "vec4 laser = laser_color * brightness;\n"
    "color = (gl_FrontFacing ? front_color + laser: back_color);\n"
"}\n";

static void draw_cube(GL gl) {
    if (!cubemap_loaded) {
        cubemap_texture_id = load_cubemap(gl);
        cubemap_loaded = true;
    }
    
    static GLuint vao = 0;
    static GLuint program_id;
    static GLuint vertex_buffer_id;
    static GLuint element_buffer_id;
    static GLint view_matrix_id;
    static GLint proj_matrix_id;
    static GLint eye_id;
    static GLint cubemap_uniform_id;
    static GLint direction_id;
    static GLint color_id;
    static GLint radius_id;
    
    static int vertex_count = get_cube_vertex_count();
    static int stride = get_cube_vertex_stride();
    static int element_count = get_cube_element_count();
    static const GLushort* cube_elements = get_cube_elements();
    
    if (vao == 0) {
        program_id = make_program(gl, cube_vs_source, cube_fs_source);
        
        view_matrix_id = gl.GetUniformLocation(program_id, "view_matrix");
        proj_matrix_id = gl.GetUniformLocation(program_id, "proj_matrix");
        eye_id = gl.GetUniformLocation(program_id, "eye");
        cubemap_uniform_id = gl.GetUniformLocation(program_id, "cubemap");
        
        direction_id = gl.GetUniformLocation(program_id, "laser_direction");
        color_id = gl.GetUniformLocation(program_id, "laser_color");
        radius_id = gl.GetUniformLocation(program_id, "laser_radius");
        
        gl.GenVertexArrays(1, &vao);
        gl.GenBuffers(1, &vertex_buffer_id);
        gl.GenBuffers(1, &element_buffer_id);
        
        gl.BindVertexArray(vao);
        
        gl.BindBuffer(GL_ELEMENT_ARRAY_BUFFER, element_buffer_id);
        gl.BufferData(
            GL_ELEMENT_ARRAY_BUFFER, element_count * sizeof(GLushort),
            cube_elements , GL_STATIC_DRAW
        );
        
        gl.BindBuffer(GL_ARRAY_BUFFER, vertex_buffer_id);
        
        gl.VertexAttribPointer(
            vertex_position_index,
            3,
            GL_FLOAT,
            false,
            stride,
            (void*)0
        );
        gl.VertexAttribPointer(
            vertex_normal_index,
            3,
            GL_FLOAT,
            false,
            stride,
            (void*)(intptr_t)get_cube_normal_offset()
        );
        gl.EnableVertexAttribArray(vertex_position_index);
        gl.EnableVertexAttribArray(vertex_normal_index);
        
        PANIC_IF_GL_ERROR(gl);
    }
    
    gl.UseProgram(program_id);
    
    gl.ActiveTexture(GL_TEXTURE0);
    gl.BindTexture(GL_TEXTURE_CUBE_MAP, cubemap_texture_id);
    gl.Uniform1i(cubemap_uniform_id, 0);
    
    gl.UniformMatrix4fv(view_matrix_id, 1, false, &view[0][0]);
    gl.UniformMatrix4fv(proj_matrix_id, 1, false, &projection[0][0]);
    gl.Uniform3fv(eye_id, 1, &eye[0]);
    
    // Make the laser black if it's off so it adds no light in the shader.
    gl.Uniform3fv(direction_id, 1, &laser_direction[0]);
    gl.Uniform4fv(color_id, 1,
        &glm::vec4(0, laser_on ? 1.0f : 0.0f, 0, laser_on ? 0.6f : 0.0f)[0]
    );
    gl.Uniform1f(radius_id, laser_beam_radius);
    
    gl.BindVertexArray(vao);
    
    gl.BindBuffer(GL_ARRAY_BUFFER, vertex_buffer_id);
    gl.BufferData(
        GL_ARRAY_BUFFER,
        stride * vertex_count,
        update_cube_vertices(),
        GL_DYNAMIC_DRAW
    );
    
    gl.DrawElements(GL_TRIANGLES, element_count, GL_UNSIGNED_SHORT, (void*)0);
    gl.BindVertexArray(0);
    gl.BindTexture(GL_TEXTURE_CUBE_MAP, 0);
    
    PANIC_IF_GL_ERROR(gl);
}

/*  Part 6: Interface and main loop.
 */
static void initialize_beads_and_cube(void) {
    reset_cube();
    
    auto bead_list = generate_face_bead_list();
    
    for (auto& bead : bead_list) {
        auto err = add_bead(
            bead.position[0], bead.position[1], bead.position[2],
            bead.color[0], bead.color[1], bead.color[2]
        );
        if (err != 0) {
            fprintf(stderr, "error attempting to add bead %s\n", strerror(err));
        }
    }
}

static bool handle_controls() {
    static bool orbit_mode = true;
    static bool w, a, s, d, q, e, space;
    static float theta = 1.5707f, phi = 1.8f, radius = 2.0f;
    static float mouse_x, mouse_y;
    
    bool no_quit = true;
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        switch (event.type) {
          default:
          break; case SDL_KEYDOWN:
            switch (event.key.keysym.scancode) {
              default:
              break; case SDL_SCANCODE_W: case SDL_SCANCODE_I: w = true;
              break; case SDL_SCANCODE_A: case SDL_SCANCODE_J: a = true;
              break; case SDL_SCANCODE_S: case SDL_SCANCODE_K: s = true;
              break; case SDL_SCANCODE_D: case SDL_SCANCODE_L: d = true;
              break; case SDL_SCANCODE_Q: case SDL_SCANCODE_U: q = true;
              break; case SDL_SCANCODE_E: case SDL_SCANCODE_O: e = true;
              break; case SDL_SCANCODE_SPACE: space = true;
              
              break; case SDL_SCANCODE_X: case SDL_SCANCODE_COMMA:
                orbit_mode = !orbit_mode;
              break; case SDL_SCANCODE_F: case SDL_SCANCODE_H:
                if (event.key.keysym.mod & KMOD_SHIFT) {
                    bead_debug = true;
                    beads_drawn = true;
                } else {
                    beads_drawn = !beads_drawn;
                    bead_debug = false;
                }
              break; case SDL_SCANCODE_R:
                if (event.key.keysym.mod & KMOD_SHIFT) {
                    initialize_beads_and_cube();
                }
              break; case SDL_SCANCODE_TAB: case SDL_SCANCODE_RETURN:
                jolt(0.6f * cosf(theta), 4.5f, 0.6f * sinf(theta));
              break; case SDL_SCANCODE_SLASH: case SDL_SCANCODE_P:
                show_controls();
            }
            
          break; case SDL_KEYUP:
            switch (event.key.keysym.scancode) {
              default:
              break; case SDL_SCANCODE_W: case SDL_SCANCODE_I: w = false;
              break; case SDL_SCANCODE_A: case SDL_SCANCODE_J: a = false;
              break; case SDL_SCANCODE_S: case SDL_SCANCODE_K: s = false;
              break; case SDL_SCANCODE_D: case SDL_SCANCODE_L: d = false;
              break; case SDL_SCANCODE_Q: case SDL_SCANCODE_U: q = false;
              break; case SDL_SCANCODE_E: case SDL_SCANCODE_O: e = false;
              break; case SDL_SCANCODE_SPACE: space = false;
            }
          break; case SDL_MOUSEWHEEL:
            phi += (orbit_mode ? 1 : -1) * event.wheel.y * 0.04f;
            theta += (orbit_mode ? 1 : -1) * event.wheel.x * 0.04f;
          break; case SDL_MOUSEBUTTONDOWN: case SDL_MOUSEBUTTONUP:
            mouse_x = event.button.x;
            mouse_y = event.button.y;
            laser_on = event.type == SDL_MOUSEBUTTONDOWN;
          break; case SDL_MOUSEMOTION:
            mouse_x = event.motion.x;
            mouse_y = event.motion.y;
          break; case SDL_WINDOWEVENT:
            if (event.window.event == SDL_WINDOWEVENT_SIZE_CHANGED ||
                event.window.event == SDL_WINDOWEVENT_RESIZED) {
                
                screen_x = event.window.data1;
                screen_y = event.window.data2;
            }
          break; case SDL_QUIT:
            no_quit = false;
        }
    }
    
    glm::vec3 forward_normal_vector(
        sinf(phi) * cosf(theta),
        cosf(phi),
        sinf(phi) * sinf(theta)
    );
    
    if (orbit_mode) {
        theta += (a * 4e-2) - (d * 4e-2);
        phi += (e * 3e-2) - (q * 3e-2);
        radius += (s * 5e-2) - (w * 5e-2);
        
        radius = clamp_float(radius, 1.25, 6.0);
        
        float const* center_ptr = get_camera_center();
        glm::vec3 center(center_ptr[0], center_ptr[1], center_ptr[2]);
        
        eye = center - radius * forward_normal_vector;
        
        view = glm::lookAt(eye, center, glm::vec3(0,1,0));
    } else {
        // Free-camera mode.
        auto right_vector = glm::cross(forward_normal_vector, glm::vec3(0,1,0));
        right_vector = glm::normalize(right_vector);
        auto up_vector = glm::cross(right_vector, forward_normal_vector);
        
        eye += 5e-2f * right_vector * (float)(d - a);
        eye += 5e-2f * forward_normal_vector * (float)(w - s);
        eye += 5e-2f * up_vector * (float)(e - q);
        
        if (space) {
            theta += 1e-4 * (mouse_x - screen_x*0.5f);
            phi +=   1e-4 * (mouse_y - screen_y*0.5f);
        }
        
        view = glm::lookAt(eye, eye+forward_normal_vector, glm::vec3(0,1,0));
    }
    phi = clamp_float(phi, 0.01f, 3.13f);
    
    projection = glm::perspective(
        fovy_radians,
        float(screen_x)/screen_y,
        near_plane,
        far_plane
    );
    
    // Invert the mouse coordinates (in device coordinates) back into
    // world coordinates to figure out the laser direction. The laser
    // position is just where the camera/eye is.
    float y_plane_radius = tanf(fovy_radians / 2.0f);
    float x_plane_radius = y_plane_radius * screen_x / screen_y;
    float mouse_vcs_x = x_plane_radius * (2.0f * mouse_x / screen_x - 1.0f);
    float mouse_vcs_y = y_plane_radius * (1.0f - 2.0f * mouse_y / screen_y);
    glm::vec4 mouse_vcs(mouse_vcs_x, mouse_vcs_y, -1.0f, 1.0f);
    glm::vec4 mouse_wcs = glm::inverse(view) * mouse_vcs;
    laser_direction = mouse_wcs - glm::vec4(eye, 1.0f);
    
    set_lamp(
        0,
        eye[0], eye[1], eye[2],
        laser_direction[0], laser_direction[1], laser_direction[2],
        laser_on ? laser_heat_per_second : 0.0f,
        laser_beam_radius
    );
    
    return no_quit;
}

int main(int, char** argv) {
    argv0 = argv[0];
    
    OpenGL_Functions gl;
    gl.Enable(GL_BLEND);
    gl.BlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    gl.Enable(GL_CULL_FACE);
    gl.Enable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
    gl.ClearColor(0, 0, 0, 1);
    
    bool no_quit = true;
    int frames = 0;
    uint32_t last_fps_print_time = 0;
    
    initialize_beads_and_cube();
    
    while (no_quit) {
        // Update physics 250 times per second (in 50 batches of 5 updates).
        static uint32_t previous_update = 0;
        uint32_t current_ticks = SDL_GetTicks();
        if (current_ticks >= previous_update + 20) {
            for (int i = 0; i < 5; ++i) tick(0.004f);
            previous_update += 20;
            if (previous_update + 20 < current_ticks) {
                previous_update = current_ticks;
            }
        }
        
        // Show FPS twice per second.
        ++frames;
        if (current_ticks >= last_fps_print_time + 500) {
            fprintf(stderr, "%i FPS\n", frames * 2);
            frames = 0;
            last_fps_print_time += 500;
            if (last_fps_print_time + 500 < current_ticks) {
                last_fps_print_time = current_ticks;
            }
        }
        
        // Update the camera and draw the stuff.
        no_quit = handle_controls();
        gl.Viewport(0, 0, screen_x, screen_y);
        
        gl.Clear(GL_COLOR_BUFFER_BIT);
        draw_skybox(gl);
        
        gl.CullFace(GL_FRONT);
        draw_cube(gl);
        gl.CullFace(GL_BACK);
        
        gl.Clear(GL_DEPTH_BUFFER_BIT);
        gl.Enable(GL_DEPTH_TEST);
        draw_beads(gl);
        gl.Disable(GL_DEPTH_TEST);
        
        draw_cube(gl);
        
        SDL_GL_SwapWindow(window);
        PANIC_IF_GL_ERROR(gl);
        
        static bool showed_controls = false;
        if (!showed_controls) {
            showed_controls = true;
            show_controls();
        }
    }
}

