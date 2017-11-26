/*    Jiggle physics engine for Jelly McJelloFace (David Akeley 2017)
 *  
 *    The game simulates a 1x1x1 jello cube with beads embedded inside of it.
 *  
 *  This file implements the backend of the game: the physics simulation and
 *  functions  for calculating and exporting geometry for the game client to
 *  render. The client interacts with the physics portion of this code using
 *  functions  like  tick, add_bead, set_lamp; the client gets geometry data
 *  describing the shape of the surface of the cube and the positions/colors
 *  of  the  beads  using  the functions implemented towards the end of this
 *  file. We are careful to ensure CCW winding for all triangles generated.
 *  
 *  The jello cube is made out of a 3D grid of (GRID_WIDTH  x  GRID_WIDTH  x
 *  GRID_WIDTH)  initially  cubic  cells  and (GRID_WIDTH+1 x GRID_WIDTH+1 x
 *  GRID_WIDTH+1) nodes. The game  will  simulate  both  the  spring  forces
 *  between  adjacent  and  diagonal  nodes,  and  pressure forces each cell
 *  exerts on the eight nodes at its corners: this pressure force depends on
 *  the  difference  between the volume and rest volume of a cell. Each node
 *  has both a "grid coordinate" that specifies its logical position in  the
 *  grid  as well as a "world coordinate" that specifies its actual position
 *  in the world. The grid coordinate determines the node's neighbors  while
 *  the  world  coordinate is used for physics and rendering. I'll try to be
 *  clear which I mean while documenting. Example: for a 10x10x10 grid,  the
 *  node  with  grid coordinate (3, 4, 5) will start with a world coordinate
 *  of (0.3, 0.4, 0.5), since the cube is always  normalized  to  be  1x1x1.
 *  Sometimes  I'll also speak of a cell's grid coordinate: this is the grid
 *  coordinate of its corner with lowest x, y, z e.g. cell (0, 0, 0) has the
 *  nodes  with  grid  coordinates (0, 0, 0), (0, 0, 1) ... (1, 1, 1) as its
 *  corners.
 *  
 *  Each bead is placed  into  a  single  cell  and  assigned  a  normalized
 *  coordinate  within that cell. The beads will be drawn as colored spheres
 *  and will be seen following the deformation of the jello  cube  they  are
 *  placed  in.  The position of a bead is the weighted average of the world
 *  coordinates of the 8 nodes at the corners of the cell the bead is placed
 *  in,  with the weights determined by the normalized coordinate (blend) of
 *  the bead. They are weightless  and  don't  affect  the  physics  of  the
 *  jello.
 *  
 *  There's also a temperature component of the  simulation.  This  isn't  a
 *  full temperature simulation in the sense that temperature doesn't spread
 *  from node-to-node. It's just a value  manipulated  by  directional  heat
 *  lamps  placed  by the user. The temperature of a node is translated to a
 *  strength value from 0 to 1 for the node. This strength value is used  to
 *  scale  the spring forces between nodes (with the force between two nodes
 *  multiplied by the strengths of the  two  endpoint  nodes),  but  has  no
 *  effect on the cell pressure. As temperature increases, the strength will
 *  decrease and the liquid-like behavior of the cell pressure  forces  will
 *  dominate.
 *  
 *  Various dampening forces are applied in the  program.  These  aren't  so
 *  well  documented  because  I  added  them  in  a  panic  trying  to stop
 *  mysterious oscillations that were tearing my cube apart. I'm  still  not
 *  happy with how the jello is more "twitchy" and less jelly-like.
 */

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include "vec3.h"
#include "jellymjf.h"

// GRID_WIDTH can't be more than 100 or so due to WebGL elements being 16-bit.
#ifndef GRID_WIDTH
#define GRID_WIDTH 15
#endif

#ifndef MAX_BEADS 
#define MAX_BEADS 8192
#endif

#ifndef MAX_LAMPS
#define MAX_LAMPS 1
#endif

static const float jolt_ceiling = 0.1f;
static const float ambient_temperature = 20.0f;
static const float node_mass = 1.0f;
static const float gravity = 1.5f;
static const float dampening_per_second = 0.375f;
static const float cooling_per_second = 0.93f;
static const float friction_per_second = 0.1f;
// Those "per_second" constants are all scale factors.
// Dampening is the amount velocity is scaled by (to 0) per second.
// Cooling is the amount heat is scaled by (to ambient) per second.
// Friction is the amount floor-touching nodes have their velocity scaled.

// spring constants and lengths for edge and diagonal springs.
#define BASE_K 1000.0f
typedef struct spring_spec {
    float k, length, max_force;
} SpringSpec;
static const SpringSpec edge_spring = {
    BASE_K,
    1.0f/GRID_WIDTH,
    1.0f/GRID_WIDTH * BASE_K
};
static const SpringSpec face_spring = {
    BASE_K * 2.0f,
    1.414214f/GRID_WIDTH,
    1.414214f/GRID_WIDTH * BASE_K * 2.0f
};
static const SpringSpec inner_spring = {
    BASE_K * 3.0f,
    1.732051f/GRID_WIDTH,
    1.732051f/GRID_WIDTH * BASE_K * 3.0f
};

// The vertical springs will be just like the edge springs, except that their
// rest length is longer so that the cube starts out supporting its own weight.
// This array is filled in the reset function.
static SpringSpec vertical_springs[GRID_WIDTH];

static void set_vertical_springs() {
    for (int i = 0; i < GRID_WIDTH; ++i) {
        float force_needed = (GRID_WIDTH-i) * node_mass * gravity;
        float displacement_needed = force_needed / edge_spring.k;
        vertical_springs[i] = (SpringSpec) {
            edge_spring.k,
            edge_spring.length + displacement_needed,
            edge_spring.max_force
        };
    }
}

/*  Calculate a node's expected strength given its temperature.
 *  Strength will be a value from 0 to 1 used to scale the spring forces.
 */
static inline float strength(float temp) {
    return clamp_float(1.0f - 0.01f*(temp-25.0f), 0.3f, 1.0f);
}

/*  Calculate a cell's pressure given its volume */
static inline float pressure(float volume) {
    float stretch = volume * (GRID_WIDTH*GRID_WIDTH*GRID_WIDTH);
    return clamp_float(1.0f-stretch, -1.0f, 1.0f) * 10000.0f;
}

typedef struct node {
    Vec3 force, velocity, prev_position, position;
    float temperature, strength, lowest_strength, mass;
} Node;

typedef struct bead {
    int cell_x, cell_y, cell_z; // Grid coordinates of the cell the bead is in.
    Vec3 blend;                 // Normalized coordinate in the cell.
    Vec3 color;
} Bead;

typedef struct lamp {
    Vec3 position;
    Vec3 normalized_direction;
    float heat_per_second, beam_radius;
} Lamp;

static Node node_array[GRID_WIDTH+1][GRID_WIDTH+1][GRID_WIDTH+1]; // [x][y][z].
static Bead bead_array[MAX_BEADS];
static int bead_count = 0;
static Lamp lamp_array[MAX_LAMPS];

int get_bead_count(void) { return bead_count; }
int get_max_beads(void) { return MAX_BEADS; }
int get_max_lamps(void) { return MAX_LAMPS; }

/*  Add a bead with "original" world coordinate xyz and float color rgb.
 *  The coordinate is "original" in the sense that this is  where  the  bead
 *  would be in the original jello cube before it's deformed. If the cube is
 *  already deformed, the bead will appear wherever the point (x,y,z) in the
 *  original cube has deformed to.
 *      Returns:
 *  0 on success
 *  ENOMEM if the bead array is full
 *  EDOM if the coordinates are not in the 1x1x1 cube.
 */
int add_bead(float x, float y, float z, float r, float g, float b) {
    if (bead_count >= MAX_BEADS) return ENOMEM;
    if (x < 0 || x > 1) return EDOM;
    if (y < 0 || y > 1) return EDOM;
    if (z < 0 || z > 1) return EDOM;
    if (!is_real(vec3(x, y, z))) return EDOM;
    
    Vec3 floaty_grid_coords = scale(GRID_WIDTH, vec3(x, y, z));
    int cell_x = (int)X(floaty_grid_coords);
    int cell_y = (int)Y(floaty_grid_coords);
    int cell_z = (int)Z(floaty_grid_coords);
    
    float blend_x = fmodf(X(floaty_grid_coords), 1.0f);
    float blend_y = fmodf(Y(floaty_grid_coords), 1.0f);
    float blend_z = fmodf(Z(floaty_grid_coords), 1.0f);
    
    // Deal with float roundoff or case when input = 1.0.
    if (cell_x >= GRID_WIDTH) {
        cell_x = GRID_WIDTH - 1;
        blend_x = 1.0f;
    }
    if (cell_y >= GRID_WIDTH) {
        cell_y = GRID_WIDTH - 1;
        blend_y = 1.0f;
    }
    if (cell_z >= GRID_WIDTH) {
        cell_z = GRID_WIDTH - 1;
        blend_z = 1.0f;
    }
    
    bead_array[bead_count++] = (Bead) {
        cell_x, cell_y, cell_z,
        vec3(blend_x, blend_y, blend_z),
        vec3(r, g, b)
    };
    return 0;
}

/*  Set lamp number [i] to have position [x, y, z], point in  the  direction
 *  of the vector [dx, dy, dz], and have the specified heating power.
 *      Returns:
 *  0 on success
 *  ENOMEM if the lamp number is out of range
 *  EDOM if the direction vector is 0, or any of the vectors are not real.
 */
int set_lamp(
    int i,
    float x, float y, float z,
    float dx, float dy, float dz,
    float heat_per_second,
    float beam_radius
) {
    if (i < 0 || i >= MAX_LAMPS) return ENOMEM;
    
    float length;
    Vec3 normalized_direction = normalize_magnitude(vec3(dx, dy, dz), &length);
    if (length <= 0.0f || !is_real(vec3(dx,dy,dz)) || !is_real(vec3(x,y,z))) {
        return EDOM;
    }
    
    lamp_array[i] = (Lamp) {
        vec3(x, y, z),
        normalized_direction,
        heat_per_second,
        beam_radius
    };
    
    return 0;
}


// Blend the 8 corner positions of the cell the bead is in to get its position.
// This is a bit more expensive than I thought.
static Vec3 bead_position(Bead b) {
    const int x = b.cell_x;
    const int y = b.cell_y;
    const int z = b.cell_z;
    float x1wt = X(b.blend);
    float y1wt = Y(b.blend);
    float z1wt = Z(b.blend);
    float x0wt = 1.0f - x1wt;
    float y0wt = 1.0f - y1wt;
    float z0wt = 1.0f - z1wt;
    
    Vec3 v000 = node_array[x][y][z].position;
    Vec3 v001 = node_array[x][y][z+1].position;
    Vec3 v010 = node_array[x][y+1][z].position;
    Vec3 v011 = node_array[x][y+1][z+1].position;
    Vec3 v100 = node_array[x+1][y][z].position;
    Vec3 v101 = node_array[x+1][y][z+1].position;
    Vec3 v110 = node_array[x+1][y+1][z].position;
    Vec3 v111 = node_array[x+1][y+1][z+1].position;
    
    Vec3 vx00 = add(scale(x0wt, v000), scale(x1wt, v100));
    Vec3 vx01 = add(scale(x0wt, v001), scale(x1wt, v101));
    Vec3 vx10 = add(scale(x0wt, v010), scale(x1wt, v110));
    Vec3 vx11 = add(scale(x0wt, v011), scale(x1wt, v111));
    
    Vec3 vxy0 = add(scale(y0wt, vx00), scale(y1wt, vx10));
    Vec3 vxy1 = add(scale(y0wt, vx01), scale(y1wt, vx11));
    
    return add(scale(z0wt, vxy0), scale(z1wt, vxy1));
}

/*  Initialize the mass/spring grid.
 *  Set the vertical springs so the system starts at equillibrium with gravity.
 *  All beads are removed by setting bead count to 0.
 *  Lamps reset to produce zero heat and at a standard position.
 *  The position of each node will be set to its normalized grid coordinate,
 *  so that the cube starts as a 1x1x1 cube in world coordinates.
 *  The velocity of each node will be zeroed.
 *  The force is untouched; it will be set to zero at the start of each tick.
 *  The temperature and strength (including lowest strength) are reset.
 *  The mass of each node will be set to node_mass.
 */
void reset_cube(void) {
    bead_count = 0;
    
    set_vertical_springs();
    
    for (int i = 0; i < MAX_LAMPS; ++i) {
        lamp_array[i].position = vec3(0.0f, 0.0f, 0.0f);
        lamp_array[i].normalized_direction = vec3(1.0, 0, 0);
        lamp_array[i].heat_per_second = 0.0f;
        lamp_array[i].beam_radius = 0.0f;
    }
    
    for (int x = 0; x <= GRID_WIDTH; ++x) {
        for (int y = 0; y <= GRID_WIDTH; ++y) {
            for (int z = 0; z <= GRID_WIDTH; ++z) {
                Node* n = &node_array[x][y][z];
                n->position = scale(1.0f/GRID_WIDTH, vec3(x, y, z));
                n->prev_position = n->position;
                n->velocity = vec3(0, 0, 0);
                n->temperature = ambient_temperature;
                n->strength = strength(ambient_temperature);
                n->lowest_strength = n->strength;
                n->mass = node_mass;
            }
        }
    }
}

/*  Jolt the jello cube by adding the specified  velocity  vector  to  nodes
 *  touching the floor (actually, below the threshold jolt_ceiling).
 */
void jolt(float dx, float dy, float dz) {
    for (int x = 0; x <= GRID_WIDTH; ++x) {
        for (int y = 0; y <= GRID_WIDTH; ++y) {
            for (int z = 0; z <= GRID_WIDTH; ++z) {
                Node* n = &node_array[x][y][z];
                
                float blend = Y(n->position) - jolt_ceiling;
                Vec3 new_velocity = add(n->velocity, vec3(dx, dy, dz));
                n->velocity = step_vec3(new_velocity, n->velocity, blend);
            }
        }
    }
}

// Calculate displacement vector [to] - [from].
static inline Vec3 node_sub(Node const* to, Node const* from) {
    return sub(to->position, from->position);
}

/*  Calculate the spring force between the two passed nodes and  add  it  to
 *  both nodes. Use the spring constant and expected length from SpringSpec.
 */
static inline void spring(Node* n0, Node* n1, SpringSpec spec) {
    float length;
    Vec3 unit_vector = normalize_magnitude(node_sub(n1, n0), &length);
    float kX = spec.k * (length - spec.length);
    float F = min_float(spec.max_force, kX * (n0->strength * n1->strength));
    Vec3 force = scale(F, unit_vector);
    iadd(&n0->force, force);
    isub(&n1->force, force);
}

/*  Add spring forces for the entire cube of nodes.
 *  Most of the nodes have 7 neighbors (we consider  only  nodes  with  grid
 *  coordinates  greater  than  that  of  the  current node to be neighbors,
 *  otherwise, spring forces would be double  counted).  However,  those  on
 *  positive-facing faces have only 3, and those on positive edges have just
 *  1 (the one at the  most  positive  corner  has  0).  These  are  handled
 *  separately.  I  hope  this  doesn't  have too huge of an impact on cache
 *  performance.
 */
static void add_all_spring_forces(void) {
    // Most of the interior.
    for (int x = 0; x < GRID_WIDTH; ++x) {
        for (int y = 0; y < GRID_WIDTH; ++y) {
            for (int z = 0; z < GRID_WIDTH; ++z) {
                Node* n = &node_array[x][y][z];
                
                spring(n, &node_array[x][y][z+1], edge_spring);
                spring(n, &node_array[x][y+1][z], vertical_springs[y]);
                spring(n, &node_array[x+1][y][z], edge_spring);
                spring(n, &node_array[x+1][y+1][z], face_spring);
                spring(n, &node_array[x+1][y][z+1], face_spring);
                spring(n, &node_array[x][y+1][z+1], face_spring);
                spring(n, &node_array[x+1][y+1][z+1], inner_spring);
            }
        }
    }
    // +z face. Calculate only for neighbors on xy plane.
    for (int x = 0; x < GRID_WIDTH; ++x) {
        for (int y = 0; y < GRID_WIDTH; ++y) {
            int z = GRID_WIDTH;
            Node* n = &node_array[x][y][z];
            
            spring(n, &node_array[x][y+1][z], vertical_springs[y]);
            spring(n, &node_array[x+1][y][z], edge_spring);
            spring(n, &node_array[x+1][y+1][z], face_spring);
        }
    }
    // +y face. Calculate only for neighbors on xz plane.
    for (int x = 0; x < GRID_WIDTH; ++x) {
        for (int z = 0; z < GRID_WIDTH; ++z) {
            int y = GRID_WIDTH;
            Node* n = &node_array[x][y][z];
            
            spring(n, &node_array[x+1][y][z], edge_spring);
            spring(n, &node_array[x][y][z+1], edge_spring);
            spring(n, &node_array[x+1][y][z+1], face_spring);
        }
    }
    // +x face. Calculate only for neighbors on yz plane.
    for (int y = 0; y < GRID_WIDTH; ++y) {
        for (int z = 0; z < GRID_WIDTH; ++z) {
            int x = GRID_WIDTH;
            Node* n = &node_array[x][y][z];
            
            spring(n, &node_array[x][y+1][z], vertical_springs[y]);
            spring(n, &node_array[x][y][z+1], edge_spring);
            spring(n, &node_array[x][y+1][z+1], face_spring);
        }
    }
    // Edges. Calculate spring force only for the next node on the edge.
    for (int i = 0; i < GRID_WIDTH; ++i) {
        int GW = GRID_WIDTH;
        SpringSpec vertical_spec = vertical_springs[i];
        spring(&node_array[i][GW][GW], &node_array[i+1][GW][GW], edge_spring);
        spring(&node_array[GW][i][GW], &node_array[GW][i+1][GW], vertical_spec);
        spring(&node_array[GW][GW][i], &node_array[GW][GW][i+1], edge_spring);
    }
}

/*  Add pressure from a single cell at grid coordinate (x, y, z).
 *  This function is performance-critical, so it's a bit hard to explain.
 *  The summary is that we're calculating the current volume of the deformed
 *  cube and using this to calculate pressure forces exerted on the 8 corner
 *  nodes.
 *  
 *  In this function I use a local grid coordinate system  where  coordinate
 *  000 is the node with grid coordinate x,y,z and 111 is the node with grid
 *  coordinate x+1, y+1, z+1.
 *  
 *  Step 1 is to calculate the "parallelogram vector" (pvec) for each of the
 *  12  triangular  faces  of  the  cube.  This  is the vector normal to the
 *  triangle, pointing outwards from the cube, and with a magnitude equal to
 *  twice the triangle's area (i.e. the parallelogram formed by doubling the
 *  triangle).
 *  
 *  Step 2 is to split the cube into 6 triangular pyramids (tetrahedra)  and
 *  sum  their  volumes  to  find the cube's volume. I'll use the 6 pyramids
 *  with the 6 triangles adjacent to node 000 as the pyramid bases and  node
 *  111  as  the tip of each pyramid. The (signed) volume of each pyramid is
 *  -1/6 times the base triangle's pvec dotted with the vector from  000  to
 *  111.
 *  
 *  Step 3 is to calculate the pressure given the  volume  just  calculated,
 *  find  the  force on each triangular face, and distribute that force over
 *  the nodes. The force on each face will be 1/2 times pressure  times  the
 *  pvec  of that face. Before the cube got distorted, each face was a right
 *  triangle: the node that was originally at that  triangle's  right  angle
 *  gets  1/2  of  the  force, and the other two nodes get 1/4. In reality I
 *  should calculate the torque on each corner and distribute based on that,
 *  but I think that's too computationally expensive to do here.
 */
static void add_cell_pressure(int x, int y, int z) {
    Node* n000 = &node_array[x][y][z];
    Node* n001 = &node_array[x][y][z+1];
    Node* n010 = &node_array[x][y+1][z];
    Node* n011 = &node_array[x][y+1][z+1];
    Node* n100 = &node_array[x+1][y][z];
    Node* n101 = &node_array[x+1][y][z+1];
    Node* n110 = &node_array[x+1][y+1][z];
    Node* n111 = &node_array[x+1][y+1][z+1];
    
    // Vectors from node[x][y][z] to the other 7 nodes.
    const Vec3 v000_001 = node_sub(n001, n000);
    const Vec3 v000_011 = node_sub(n011, n000);
    const Vec3 v000_010 = node_sub(n010, n000);
    const Vec3 v000_110 = node_sub(n110, n000);
    const Vec3 v000_100 = node_sub(n100, n000);
    const Vec3 v000_101 = node_sub(n101, n000);
    const Vec3 v000_111 = node_sub(n111, n000);
    
    // Paralellogram vectors for 6 triangles around node 000.
    const Vec3 pvec_000_001_011 = cross(v000_001, v000_011);
    const Vec3 pvec_000_011_010 = cross(v000_011, v000_010);
    const Vec3 pvec_000_010_110 = cross(v000_010, v000_110);
    const Vec3 pvec_000_110_100 = cross(v000_110, v000_100);
    const Vec3 pvec_000_100_101 = cross(v000_100, v000_101);
    const Vec3 pvec_000_101_001 = cross(v000_101, v000_001);
    
    // Sum of the dot products using less-than-obvious algorithm.
    Vec3 s =   elementwise_mul(v000_111, pvec_000_001_011);
    Vec3 t =   elementwise_mul(v000_111, pvec_000_011_010);
    s = add(s, elementwise_mul(v000_111, pvec_000_010_110));
    t = add(t, elementwise_mul(v000_111, pvec_000_110_100));
    s = add(s, elementwise_mul(v000_111, pvec_000_100_101));
    t = add(t, elementwise_mul(v000_111, pvec_000_101_001));
    Vec3 SS = add(s, t);
    
    const float volume = (-1.0f/6.0f) * (X(SS) + Y(SS) + Z(SS));
    const float quarter_pressure = 0.25f * pressure(volume);
    
    // Add forces onto the 3 nodes of each triangle. The right-angle node
    // (across from hypotenuse) gets 1/2 of the force and gets added first.
    // The other two nodes come next and get 1/4 each.
    Vec3 force0, force1;
    
    iadd(&n001->force, force0 = scale(quarter_pressure, pvec_000_001_011));
    iadd(&n011->force, force0 = scale(0.5f, force0));
    iadd(&n000->force, force0);
    
    iadd(&n010->force, force1 = scale(quarter_pressure, pvec_000_011_010));
    iadd(&n011->force, force1 = scale(0.5f, force1));
    iadd(&n000->force, force1);
    
    iadd(&n010->force, force0 = scale(quarter_pressure, pvec_000_010_110));
    iadd(&n110->force, force0 = scale(0.5f, force0));
    iadd(&n000->force, force0);
    
    iadd(&n100->force, force1 = scale(quarter_pressure, pvec_000_110_100));
    iadd(&n110->force, force1 = scale(0.5f, force1));
    iadd(&n000->force, force1);
    
    iadd(&n100->force, force0 = scale(quarter_pressure, pvec_000_100_101));
    iadd(&n101->force, force0 = scale(0.5f, force0));
    iadd(&n000->force, force0);
    
    iadd(&n001->force, force1 = scale(quarter_pressure, pvec_000_101_001));
    iadd(&n101->force, force1 = scale(0.5f, force1));
    iadd(&n000->force, force1);
    
    // Now add pressure force for the other 6 triangles. I hope that
    // the compiler(s) used have a good register allocation strategy.
    const Vec3 v111_001 = node_sub(n001, n111);
    const Vec3 v111_011 = node_sub(n011, n111);
    const Vec3 v111_010 = node_sub(n010, n111);
    const Vec3 v111_110 = node_sub(n110, n111);
    const Vec3 v111_100 = node_sub(n100, n111);
    const Vec3 v111_101 = node_sub(n101, n111);
    
    const Vec3 pvec_111_001_101 = cross(v111_001, v111_101);
    const Vec3 pvec_111_101_100 = cross(v111_101, v111_100);
    const Vec3 pvec_111_100_110 = cross(v111_100, v111_110);
    const Vec3 pvec_111_110_010 = cross(v111_110, v111_010);
    const Vec3 pvec_111_010_011 = cross(v111_010, v111_011);
    const Vec3 pvec_111_011_001 = cross(v111_011, v111_001);
    
    iadd(&n101->force, force0 = scale(quarter_pressure, pvec_111_001_101));
    iadd(&n001->force, force0 = scale(0.5f, force0));
    iadd(&n111->force, force0);
    
    iadd(&n101->force, force1 = scale(quarter_pressure, pvec_111_101_100));
    iadd(&n100->force, force1 = scale(0.5f, force1));
    iadd(&n111->force, force1);
    
    iadd(&n110->force, force0 = scale(quarter_pressure, pvec_111_100_110));
    iadd(&n100->force, force0 = scale(0.5f, force0));
    iadd(&n111->force, force0);
    
    iadd(&n110->force, force1 = scale(quarter_pressure, pvec_111_110_010));
    iadd(&n010->force, force1 = scale(0.5f, force1));
    iadd(&n111->force, force1);
    
    iadd(&n011->force, force0 = scale(quarter_pressure, pvec_111_010_011));
    iadd(&n010->force, force0 = scale(0.5f, force0));
    iadd(&n111->force, force0);
    
    iadd(&n011->force, force1 = scale(quarter_pressure, pvec_111_011_001));
    iadd(&n001->force, force1 = scale(0.5f, force1));
    iadd(&n111->force, force1);
    
    // I don't care if you think my function is too long. Cubes are complicated!
}

/*  Called after all forces summed up for this node. Use Euler's  method  to
 *  update  velocity  and position given the total force calculated for this
 *  node, scale the velocity by the given dampening factor, and,  for  nodes
 *  with  position y <= 0, raise them up to the floor at y=0 and scale their
 *  velocity by the given friction factor.
 *  There's also this kludgy phantom dampening force  that's  applied  where
 *  the  dot  product  is below. If the node recently made a sharp turn (>90
 *  degrees), it zeroes out its  velocity  and  sets  its  position  to  the
 *  average  of the last 3 positions calculated for it. I needed this to fix
 *  "twitchy" oscillations that have a period of  only  a  few  ticks  long.
 *  Regular  dampening forces weren't enough to fix this oscillation because
 *  their period was too short; if I could afford more ticks  per  second  I
 *  wouldn't  need  this extra dampening, but I can't afford it so I have to
 *  live with this hack.
 */
static void node_update(Node* n, float dt, float dampening, float friction) {
    Vec3 new_velocity = add(n->velocity, scale(dt/n->mass, n->force));
    new_velocity = scale(dampening, new_velocity);
    Vec3 new_position = add(n->position, scale(dt, new_velocity));
    
    // Fudge position and velocity for nodes making sharp turns (dot < 0).
    Vec3 fudge_vel = vec3(0, 0, 0);
    Vec3 fudge_pos = scale(0.3333333f,
            add(add(new_position, n->position), n->prev_position)
    );
    float DOT = dot(
        sub(new_position, n->position), sub(n->position, n->prev_position)
    );
    new_velocity = step_vec3(fudge_vel, new_velocity, DOT);
    new_position = step_vec3(fudge_pos, new_position, DOT);
    
    // Apply friction and enforce floor at y=0.
    float y = Y(new_position);
    Vec3 floor_pos = vec3(X(new_position), 0.0f, Z(new_position));
    Vec3 floor_vel = scale(friction, vec3(X(new_velocity), 0, Z(new_velocity)));
    
    // If the new position has NaNs or infinities, DON'T WRITE IT BACK!
    // Just leave the node still and hope the problem resolves itself.
    if (is_real(new_position)) {
        n->prev_position = n->position;
        n->position = step_vec3(new_position, floor_pos, -y);
        n->velocity = step_vec3(new_velocity, floor_vel, -y);
    }
}

// Update the temperature and strength of the node given lamp positions.
static void node_temperature_strength(Node* n, float dt, float decay) {
    float new_temperature = n->temperature;
    new_temperature -= (1-decay) * (new_temperature - ambient_temperature);
    
    for (int i = 0; i < MAX_LAMPS; ++i) {
        Lamp* lamp = &lamp_array[i];
        Vec3 v = lamp->normalized_direction;
        
        Vec3 node_disp = sub(n->position, lamp->position);
        Vec3 beam_projection = scale(dot(node_disp, v), v);
        Vec3 perpendicular = sub(node_disp, beam_projection);
        // perpendicular is the distance the node is from the lamp's beam.
        
        // Calculate the actual heat generated by this lamp for this node.
        // The step functions zero-out heat if the node is too far from
        // the lamp's beam, or if the node is behind the lamp.
        float r = lamp->beam_radius;
        float heat = dt * lamp->heat_per_second;
        heat = step(heat, r*r - sum_squares(perpendicular));
        heat = step(heat, dot(node_disp, beam_projection));
        
        new_temperature += heat;
    }
    
    // To simulate permanent loss-of-shape after heating, we record the
    // lowest strength this node has ever had and put a ceiling on its
    // current strength based on that lowest strength.
    float new_strength = strength(new_temperature);
    n->lowest_strength = min_float(new_strength, n->lowest_strength);
    n->strength = min_float(new_strength, 1.75f * n->lowest_strength);
    n->temperature = new_temperature;
}

static float camera_center[3];

// Tick the simulation by a time step dt.
// Also updates the recommended center position of the camera
// (average position the center node had in the last few ticks).
void tick(float dt) {
    // Reset the force vector of each node to force of gravity.
    // Recalculate temperature and strength of each node.
    float decay = expf(logf(cooling_per_second) * dt);
    for (int x = 0; x <= GRID_WIDTH; ++x) {
        for (int y = 0; y <= GRID_WIDTH; ++y) {
            for (int z = 0; z <= GRID_WIDTH; ++z) {
                Node* n = &node_array[x][y][z];
                n->force = vec3(0, -gravity * n->mass, 0);
                node_temperature_strength(n, dt, decay);
            }
        }
    }
    
    add_all_spring_forces();
    
    // Note that cell x,y,z range from 0 to GRID_WIDTH - 1.
    for (int x = 0; x < GRID_WIDTH; ++x) {
        for (int y = 0; y < GRID_WIDTH; ++y) {
            for (int z = 0; z < GRID_WIDTH; ++z) {
                add_cell_pressure(x, y, z);
            }
        }
    }
    
    float dampening = expf(logf(dampening_per_second) * dt);
    float friction = expf(logf(friction_per_second) * dt);
    for (int x = 0; x <= GRID_WIDTH; ++x) {
        for (int y = 0; y <= GRID_WIDTH; ++y) {
            for (int z = 0; z <= GRID_WIDTH; ++z) {
                node_update(&node_array[x][y][z], dt, dampening, friction);
            }
        }
    }
    
    static Vec3 samples[600];
    static int sample_count = sizeof samples / sizeof(Vec3);
    static int samples_init = 0;
    static int samples_i = 0;
    static int a = GRID_WIDTH / 2;
    static int b = (GRID_WIDTH+1) / 2;
    
    Vec3 this_tick_center = scale(0.125f,
        add(
            add(
                add(node_array[a][a][a].position,
                    node_array[a][a][b].position),
                add(node_array[a][b][a].position,
                    node_array[a][b][b].position)
            ),add(
                add(node_array[b][a][a].position,
                    node_array[b][a][b].position),
                add(node_array[b][b][a].position,
                    node_array[b][b][b].position)
            )
        )
    );
    samples[samples_i] = this_tick_center;
    samples_i++;
    if (samples_i >= sample_count) samples_i = 0;
    
    Vec3 total = vec3(0, 0, 0);
    for (int i = 0; i < sample_count; ++i) {
        if (!samples_init) samples[i] = this_tick_center;
        iadd(&total, samples[i]);
    }
    
    samples_init = 1;
    iscale(&total, 1.0f / sample_count);
    camera_center[0] = X(total);
    camera_center[1] = Y(total);
    camera_center[2] = Z(total);
}

float const* get_camera_center(void) {
    return camera_center;
}

/*  Code ahead handles exporting data from the simulation to the client that
 *  draws  the  cube and beads. Functions update and return pointers to data
 *  in a format suitable for being directly passed to glBufferData.
 */

// 3 vertices per triangle, 6 faces, 2*GRID_WIDTH*GRID_WIDTH triangles per face
#define FACE_ELEMENT_COUNT  (3 * 2 * GRID_WIDTH * GRID_WIDTH)
#define CUBE_ELEMENT_COUNT  (6 * FACE_ELEMENT_COUNT)
// 6 faces, (GRID_WIDTH+1)*(GRID_WIDTH*1) vertices per face.
// Edges and corners don't share vertices due to differing normals.
#define FACE_VERTEX_COUNT  ((GRID_WIDTH+1) * (GRID_WIDTH+1))
#define CUBE_VERTEX_COUNT  (6 * FACE_VERTEX_COUNT)

static uint16_t cube_element_array[CUBE_ELEMENT_COUNT];
static CubeVertex cube_vertex_array[CUBE_VERTEX_COUNT];
static BeadVertex bead_vertex_array[MAX_BEADS];

// Return data needed to pass vertex/element arrays to OpenGL VBO.
int get_cube_element_count(void) { return CUBE_ELEMENT_COUNT; }
int get_cube_vertex_count(void) { return CUBE_VERTEX_COUNT; }

// We are using interleaved arrays. Position data will always be at offset 0.
// These functions tell us the byte offset of the other vertex data, and the
// overall stride between one vertex and the next.
int get_cube_vertex_stride(void) { return sizeof(CubeVertex); }
int get_cube_normal_offset(void) {
    return (char*)(&cube_vertex_array[0].normal)
         - (char*)(&cube_vertex_array[0]);
}
int get_bead_vertex_stride(void) { return sizeof(BeadVertex); }
int get_bead_color_offset(void) {
    return (char*)(&bead_vertex_array[0].color)
         - (char*)(&bead_vertex_array[0]);
}

/*  Return a pointer to a  statically-allocated  array  of  16-bit  elements
 *  suitable  for use in glDrawElements in conjunction with vertex data from
 *  the next function (update_cube_vertices). The node positions  from  that
 *  function  are  connected into little triangles in the logical way; these
 *  triangles always have CCW winding when viewed from outside of the  cube.
 *  This function always returns the same data.
 */
uint16_t const* get_cube_elements(void) {
    // The six partitions of the element array corresponding to 6 cube faces.
    uint16_t* x0_face = &cube_element_array[0 * FACE_ELEMENT_COUNT];
    uint16_t* x1_face = &cube_element_array[1 * FACE_ELEMENT_COUNT];
    uint16_t* y0_face = &cube_element_array[2 * FACE_ELEMENT_COUNT];
    uint16_t* y1_face = &cube_element_array[3 * FACE_ELEMENT_COUNT];
    uint16_t* z0_face = &cube_element_array[4 * FACE_ELEMENT_COUNT];
    uint16_t* z1_face = &cube_element_array[5 * FACE_ELEMENT_COUNT];
    
    int index = 0;
    
    // Fill in the element array six squares at a time.
    // 2*3 = 6 vertices per square, and we fill the 6 faces' squares
    // in parallel.
    for (int b = 0; b < GRID_WIDTH; ++b) {
        for (int a = 0; a < GRID_WIDTH; ++a) {
            z1_face[index] = FACE_VERTEX_COUNT +
            (z0_face[index] = FACE_VERTEX_COUNT +
            (y1_face[index] = FACE_VERTEX_COUNT +
            (y0_face[index] = FACE_VERTEX_COUNT +
            (x1_face[index] = FACE_VERTEX_COUNT +
            (x0_face[index] = a + b * (GRID_WIDTH+1))))));
            ++index;
            
            z1_face[index] = FACE_VERTEX_COUNT +
            (z0_face[index] = FACE_VERTEX_COUNT +
            (y1_face[index] = FACE_VERTEX_COUNT +
            (y0_face[index] = FACE_VERTEX_COUNT +
            (x1_face[index] = FACE_VERTEX_COUNT +
            (x0_face[index] = (a+1) + (b+1) * (GRID_WIDTH+1))))));
            ++index;
            
            z1_face[index] = FACE_VERTEX_COUNT +
            (z0_face[index] = FACE_VERTEX_COUNT +
            (y1_face[index] = FACE_VERTEX_COUNT +
            (y0_face[index] = FACE_VERTEX_COUNT +
            (x1_face[index] = FACE_VERTEX_COUNT +
            (x0_face[index] = a + (b+1) * (GRID_WIDTH+1))))));
            ++index;
            
            z1_face[index] = FACE_VERTEX_COUNT +
            (z0_face[index] = FACE_VERTEX_COUNT +
            (y1_face[index] = FACE_VERTEX_COUNT +
            (y0_face[index] = FACE_VERTEX_COUNT +
            (x1_face[index] = FACE_VERTEX_COUNT +
            (x0_face[index] = a + b * (GRID_WIDTH+1))))));
            ++index;
            
            z1_face[index] = FACE_VERTEX_COUNT +
            (z0_face[index] = FACE_VERTEX_COUNT +
            (y1_face[index] = FACE_VERTEX_COUNT +
            (y0_face[index] = FACE_VERTEX_COUNT +
            (x1_face[index] = FACE_VERTEX_COUNT +
            (x0_face[index] = (a+1) + b * (GRID_WIDTH+1))))));
            ++index;
            
            z1_face[index] = FACE_VERTEX_COUNT +
            (z0_face[index] = FACE_VERTEX_COUNT +
            (y1_face[index] = FACE_VERTEX_COUNT +
            (y0_face[index] = FACE_VERTEX_COUNT +
            (x1_face[index] = FACE_VERTEX_COUNT +
            (x0_face[index] = (a+1) + (b+1) * (GRID_WIDTH+1))))));
            ++index;
        }
    }
    
    return cube_element_array;
}

/*  Recalculate the positions and normals of nodes at  the  surface  of  the
 *  cube. Always returns the same pointer to a statically-allocated array of
 *  vertex data. All triangles have CCW winding when viewed from outside the
 *  cube.
 */
CubeVertex const* update_cube_vertices(void) {
    // The six partitions of the vertex array corresponding to 6 cube faces.
    CubeVertex* x0_face = &cube_vertex_array[0 * FACE_VERTEX_COUNT];
    CubeVertex* x1_face = &cube_vertex_array[1 * FACE_VERTEX_COUNT];
    CubeVertex* y0_face = &cube_vertex_array[2 * FACE_VERTEX_COUNT];
    CubeVertex* y1_face = &cube_vertex_array[3 * FACE_VERTEX_COUNT];
    CubeVertex* z0_face = &cube_vertex_array[4 * FACE_VERTEX_COUNT];
    CubeVertex* z1_face = &cube_vertex_array[5 * FACE_VERTEX_COUNT];
    
    // First fill in the positions of the vertices, and zero
    // out the normal vector in preparation for calculating normals.
    // Note that we're working per-vertex, not per triangle, so
    // the coordinate range is 0...GRID_WIDTH inclusive, not exclusive.
    // As before we fill the 6 face partitions in parallel.
    for (int b = 0; b <= GRID_WIDTH; ++b) {
        for (int a = 0; a <= GRID_WIDTH; ++a) {
            int index = a + b*(GRID_WIDTH+1);
            
            x0_face[index].position = node_array[0][b][a].position;
            x0_face[index].normal = vec3(0,0,0);
            
            x1_face[index].position = node_array[GRID_WIDTH][a][b].position;
            x1_face[index].normal = vec3(0,0,0);
            
            y0_face[index].position = node_array[a][0][b].position;
            y0_face[index].normal = vec3(0,0,0);
            
            y1_face[index].position = node_array[b][GRID_WIDTH][a].position;
            y1_face[index].normal = vec3(0,0,0);
            
            z0_face[index].position = node_array[b][a][0].position;
            z0_face[index].normal = vec3(0,0,0);
            
            z1_face[index].position = node_array[a][b][GRID_WIDTH].position;
            z1_face[index].normal = vec3(0,0,0);
        }
    }
    
    // For each triangular face, add its normal to its 3 corners. This results
    // in inner vertices having a normal that is a blend of its neighbors,
    // though some triangles will have more influence than others because
    // I'm too cheap to normalize the normal vectors.
    for (int b = 0; b < GRID_WIDTH; ++b) {
        for (int a = 0; a < GRID_WIDTH; ++a) {
            // We're looking at the square with grid coordinates
            // (a, b) to (a+1, b+1). i00 i01 ... i11 are the indicies
            // within the face arrays corresponding to the nodes
            // with grid coordinates (a, b), (a, b+1) ... (a+1, b+1).
            // v01, v11, v10 are the differences in position between the
            // nodes with grid coordinates (a, b+1), (a+1, b+1), (a+1, b)
            // and the node with grid coordinate (a, b).
            // (These 2D grid coordinates are the 3D coordinates with the
            // constant coordinate for a face removed, e.g., (a,b) is grid
            // coordinate (a, 0, b) for y=0 face). Note that sometimes a/b
            // are reversed in order to keep the normals and front
            // (anticlockwise) face pointing the right way.
            int i00 = a     + b * (GRID_WIDTH+1);
            int i01 = a     + (b+1) * (GRID_WIDTH+1);
            int i10 = a + 1 + b * (GRID_WIDTH+1);
            int i11 = a + 1 + (b+1) * (GRID_WIDTH+1);
            
            Vec3 v01, v11, v10, normal0, normal1;
            Node* n00;
            
#define ADD_TRIANGLE_NORMALS(face) \
            normal0 = cross(v11, v01); \
            normal1 = cross(v10, v11); \
            iadd(&face[i00].normal, normal0); \
            iadd(&face[i11].normal, normal0); \
            iadd(&face[i01].normal, normal0); \
            iadd(&face[i00].normal, normal1); \
            iadd(&face[i10].normal, normal1); \
            iadd(&face[i11].normal, normal1)
            
            n00 = &node_array[0][b][a];
            v01 = node_sub(&node_array[0][b+1][a], n00);
            v11 = node_sub(&node_array[0][b+1][a+1], n00);
            v10 = node_sub(&node_array[0][b][a+1], n00);
            ADD_TRIANGLE_NORMALS(x0_face);
            
            n00 = &node_array[GRID_WIDTH][a][b];
            v01 = node_sub(&node_array[GRID_WIDTH][a][b+1], n00);
            v11 = node_sub(&node_array[GRID_WIDTH][a+1][b+1], n00);
            v10 = node_sub(&node_array[GRID_WIDTH][a+1][b], n00);
            ADD_TRIANGLE_NORMALS(x1_face);
            
            n00 = &node_array[a][0][b];
            v01 = node_sub(&node_array[a][0][b+1], n00);
            v11 = node_sub(&node_array[a+1][0][b+1], n00);
            v10 = node_sub(&node_array[a+1][0][b], n00);
            ADD_TRIANGLE_NORMALS(y0_face);
            
            n00 = &node_array[b][GRID_WIDTH][a];
            v01 = node_sub(&node_array[b+1][GRID_WIDTH][a], n00);
            v11 = node_sub(&node_array[b+1][GRID_WIDTH][a+1], n00);
            v10 = node_sub(&node_array[b][GRID_WIDTH][a+1], n00);
            ADD_TRIANGLE_NORMALS(y1_face);
            
            n00 = &node_array[a][b][0];
            v01 = node_sub(&node_array[b+1][a][0], n00);
            v11 = node_sub(&node_array[b+1][a+1][0], n00);
            v10 = node_sub(&node_array[b][a+1][0], n00);
            ADD_TRIANGLE_NORMALS(z0_face);
            
            n00 = &node_array[a][b][GRID_WIDTH];
            v01 = node_sub(&node_array[a][b+1][GRID_WIDTH], n00);
            v11 = node_sub(&node_array[a+1][b+1][GRID_WIDTH], n00);
            v10 = node_sub(&node_array[a+1][b][GRID_WIDTH], n00);
            ADD_TRIANGLE_NORMALS(z1_face);
        }
    }
    
    return cube_vertex_array;
}

/*  Refill the bead vertex array with the latest data.  Always  returns  the
 *  same pointer to a static array of bead vertices.
 */
BeadVertex const* update_bead_vertices(void) {
    for (int i = 0; i < bead_count; ++i) {
        bead_vertex_array[i].position = bead_position(bead_array[i]);
        bead_vertex_array[i].color = bead_array[i].color;
    }
    return bead_vertex_array;
}

/*  Alternate bead arrays that are filled with 1 bead per  node,  with  that
 *  bead  colored  based  on the node's temperature. There's an additional 4
 *  beads to help indicate the world coordinate axes.
 */
#define DEBUG_BEAD_COUNT  ((GRID_WIDTH+1)*(GRID_WIDTH+1)*(GRID_WIDTH+1) + 4)
int get_debug_bead_count(void) {
    return DEBUG_BEAD_COUNT;
}
BeadVertex const* update_debug_bead_vertices(void) {
    static BeadVertex array[DEBUG_BEAD_COUNT];
    array[0].position = vec3(0, 0, 0);
    array[0].color = vec3(1, 1, 1);
    array[1].position = vec3(1.2f, 0, 0);
    array[1].color = vec3(1, 0, 0);
    array[2].position = vec3(0, 1.2f, 0);
    array[2].color = vec3(0, 1, 0);
    array[3].position = vec3(0, 0, 1.2f);
    array[3].color = vec3(0.3f, 0.6f, 1);
    int index = 4;
    for (int x = 0; x <= GRID_WIDTH; ++x) {
        for (int y = 0; y <= GRID_WIDTH; ++y) {
            for (int z = 0; z <= GRID_WIDTH; ++z, ++index) {
                array[index].position = node_array[x][y][z].position;
                float temperature = node_array[x][y][z].temperature;
                float r = (temperature - ambient_temperature)*.01f;
                float g = 0.4f;
                float b = 1.0f - (temperature - ambient_temperature)*.01f;
                
                if (r < 0.0f) r = 0.0f;
                if (r > 1.0f) r = 1.0f;
                if (b < 0.0f) b = 0.0f;
                if (b > 1.0f) b = 1.0f;
                
                array[index].color = vec3(r, g, b);
            }
        }
    }
    return array;
}

