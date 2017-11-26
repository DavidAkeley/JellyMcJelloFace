// See jellymjf.c for documentation.
#ifndef JELLYMCJELLOFACE_JELLYMJF_H_
#define JELLYMCJELLOFACE_JELLYMJF_H_
#include "vec3.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cube_vertex {
    Vec3 position;
    Vec3 normal;
} CubeVertex;

typedef struct bead_vertex {
    Vec3 position;
    Vec3 color;
} BeadVertex;

int get_bead_count(void);
int get_max_beads(void);
int get_max_lamps(void);

int add_bead(float x, float y, float z, float r, float g, float b);

int set_lamp(int i,
             float x, float y, float z,
             float dx, float dy, float dz,
             float heat_per_second,
             float beam_radius
            );

void reset_cube(void);
void jolt(float dx, float dy, float dz);
void tick(float dt);

float const* get_camera_center(void);
int get_cube_element_count(void);
int get_cube_vertex_count(void);
int get_cube_vertex_stride(void);
int get_cube_normal_offset(void);
int get_bead_vertex_stride(void);
int get_bead_color_offset(void);

uint16_t const* get_cube_elements(void);
CubeVertex const* update_cube_vertices(void);
BeadVertex const* update_bead_vertices(void);

int get_debug_bead_count(void);
BeadVertex const* update_debug_bead_vertices(void);

#ifdef __cplusplus
}
#endif

#endif

