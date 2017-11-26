/*  Functions for generating a list of beads at runtime that approximate the
 *  shape of a face.
 */
#ifndef JELLYMCJELLOFACE_BEADFACE_H_
#define JELLYMCJELLOFACE_BEADFACE_H_

#include <vector>
#include <math.h>
#include <time.h>
#include <random>

#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

namespace beadface {

struct Bead {
    glm::vec3 position, color;
    Bead(glm::vec3 p, glm::vec3 c) : position(p), color(c) { }
    Bead(glm::vec4 p, glm::vec3 c) : position(p), color(c) { }
};

using BeadList = std::vector<Bead>;

static std::mt19937 random(time(nullptr));
static inline float random_float() {
    return random() * (1.0 / 4294967296.0);
}

/*  Append a list of [bead_count] beads to the  list  of  beads  (using  the
 *  specified  color). The list of beads forms a circle with radius 1 in the
 *  xz plane before being transformed by the matrix M.
 */
static inline void add_bead_loop(
    BeadList* list, glm::mat4 M, glm::vec3 color, int bead_count
) {
    float pi_portion = 2 * M_PI / bead_count;
    for (int i = 0; i < bead_count; ++i) {
        auto z_wt = sinf(pi_portion * i);
        auto x_wt = cosf(pi_portion * i);
        list->emplace_back(M * glm::vec4(x_wt, 0, z_wt, 1), color);
    }
}

static inline void hair_random_walk_step(glm::vec4* position);

/*  Add a hair by starting a random walk at the origin of the  M  coordinate
 *  system. The random walk moves along the x, z, and positive y axis of the
 *  M system, leaving a bead at each step. We also add a dyed tip.
 */
static inline void add_hair(
    BeadList* list, glm::mat4 M, glm::vec3 hair_color, glm::vec3 tip_color
) {
    glm::vec4 position(0, 0, 0, 1);
    for (int i = 0; i < 16; ++i) {
        hair_random_walk_step(&position);
        list->emplace_back(M * position, hair_color);
    }
    // Define new coordinate system for the dyed tip.
    auto M_tip = glm::translate(M, glm::vec3(position));
    position = glm::vec4(0, 0, 0, 1);
    for (int i = 0; i < 6; ++i) {
        hair_random_walk_step(&position);
        list->emplace_back(M_tip * position, tip_color);
    }
}

static inline void hair_random_walk_step(glm::vec4* position) {
    const float step = 0.02f;
    auto& p = *position;
    switch (random() & 7) {
      default:
        p[1] += step;
      break; case 0:
        p[0] += step;
      break; case 1:
        p[0] -= step;
      break; case 2:
        p[2] += step;
      break; case 3:
        p[2] -= step;
    }
}

/*  Add an ear at the origin of the M system. The ear will extend in the  +x
 *  direction only. An earring is included as a child object.
 */
static inline void add_ear(
    BeadList* list, glm::mat4 M, glm::vec3 ear_color, glm::vec3 earring_color
) {
    const float x_radius = 0.07f, y_radius = 0.1f;
    
    // Center of the ear is offset from M's origin so that it's in +x only.
    auto M_ear = glm::translate(M, glm::vec3(x_radius, 0, 0));
    
    // Two loops representing the ear are ellipsoids in xy plane.
    auto M_ear_loop = glm::rotate(M_ear, float(M_PI) / 2, glm::vec3(1, 0, 0));
    M_ear_loop = glm::scale(M_ear_loop, glm::vec3(x_radius, 0, y_radius));
    add_bead_loop(list, M_ear_loop, ear_color, 32);
    M_ear_loop = glm::scale(M_ear_loop, glm::vec3(0.5f, 0, 0.5f));
    add_bead_loop(list, M_ear_loop, ear_color, 16);
    
    // Earring is in yz plane instead.
    auto M_earring = glm::translate(M_ear, glm::vec3(0, -y_radius, 0));
    M_earring = glm::rotate(M_earring, float(M_PI) / 2, glm::vec3(0, 0, 1));
    M_earring = glm::scale(M_earring, glm::vec3(y_radius/4, 0, y_radius/4));
    add_bead_loop(list, M_earring, earring_color, 8);
}

/*  Draw an eye that's a disk  in  the  xy  plane  of  M  (centered  at  M's
 *  origin).
 */
static inline void add_eye(
    BeadList* list, glm::mat4 M, glm::vec3 eye_color
) {
    auto M_eye = glm::rotate(M, float(M_PI)/2, glm::vec3(1, 0, 0));
    auto M5 = glm::scale(M_eye, glm::vec3(0.05f, 0, 0.05f));
    auto M4 = glm::scale(M_eye, glm::vec3(0.04f, 0, 0.04f));
    auto M3 = glm::scale(M_eye, glm::vec3(0.03f, 0, 0.03f));
    auto M2 = glm::scale(M_eye, glm::vec3(0.02f, 0, 0.02f));
    auto M1 = glm::scale(M_eye, glm::vec3(0.01f, 0, 0.01f));
    add_bead_loop(list, M1, glm::vec3(0, 0, 0), 4);
    add_bead_loop(list, M2, glm::vec3(0, 0, 0), 8);
    add_bead_loop(list, M3, eye_color, 12);
    add_bead_loop(list, M4, glm::vec3(1, 1, 1), 16);
    add_bead_loop(list, M5, glm::vec3(1, 1, 1), 20);
}

} // end namespace

static inline std::vector<beadface::Bead> generate_face_bead_list() {
    using namespace beadface;
    std::vector<beadface::Bead> result;
    glm::mat4 I(1.0f);
    
    float head_xz_radius = 0.25f;
    auto M_head = glm::translate(I, glm::vec3(0.5f, 0.42f, 0.375f));
    auto skin_color = glm::vec3(0.7f, 0.7f, 0.5f);
    
    // Draw the ellipsoid head and hair extending in the +z direction.
    for (int i = -10; i <= 10; ++i) {
        float y = i * 0.03f; 
        float r = head_xz_radius * 3.1623f * sqrtf(0.1f - y*y);
        auto M_loop = glm::translate(M_head, glm::vec3(0, y, 0));
        M_loop = glm::scale(M_loop, glm::vec3(r, r, r));
        add_bead_loop(&result, M_loop, skin_color, 48);
        
        if (i < -6) continue; // Don't add hair below this level.
        
        for (float angle = -1.25f; angle <= 1.25f; angle += 0.25f) {
            glm::vec4 loop_coordinate(sinf(angle), 0, cosf(angle), 1);
            auto M_hair = glm::translate(I, glm::vec3(M_loop*loop_coordinate));
            M_hair = glm::rotate(M_hair, angle, glm::vec3(0, 1, 0));
            M_hair = glm::rotate(M_hair, 1.7f - i*.1f, glm::vec3(1, 0, 0));
            
            auto gray = 0.2f + 0.2f * random_float();
            glm::vec3 hair_color(gray, gray, gray);
            glm::vec3 tip_color(0.9f, 0.9f, 0.9f - 0.3f*random_float());
            
            add_hair(&result, M_hair, hair_color, tip_color);
        }
    }
    
    // Draw the two ears.
    auto M_ear1 = glm::translate(M_head, glm::vec3(head_xz_radius, 0, 0));
    add_ear(&result, M_ear1, skin_color, glm::vec3(0.1f, 0.8f, 0.1f));
    
    auto M_ear2 = glm::translate(M_head, glm::vec3(-head_xz_radius, 0, 0));
    M_ear2 = glm::scale(M_ear2, glm::vec3(-1, 1, 1));
    add_ear(&result, M_ear2, skin_color, glm::vec3(0.8f, 0.1f, 0.1f));
    
    // Draw the two eyes. Position them by moving them to the front of the
    // head and then rotating to position them to the two sides.
    auto eye_color = glm::vec3(.2, .4+random_float()*.3, random_float());
    auto M_eye1 = glm::rotate(M_head, 0.3f, glm::vec3(0, 1, 0));
    M_eye1 = glm::translate(M_eye1, glm::vec3(0, 0, -head_xz_radius - 0.01f));
    add_eye(&result, M_eye1, eye_color);
    auto M_eye2 = glm::rotate(M_head, -0.3f, glm::vec3(0, 1, 0));
    M_eye2 = glm::translate(M_eye2, glm::vec3(0, 0, -head_xz_radius - 0.01f));
    add_eye(&result, M_eye2, eye_color);
    
    // Add a mouth. This is hard-coded due to laziness.
    for (float f = -3.0f; f <= 3.0f; f += 0.5f) {
        glm::vec4 coordinate(f * 0.04f,
                             0.02f*sinf(f - 0.7f) - 0.13f,
                             -0.24f,
                             1);
        result.emplace_back(glm::vec3(M_head*coordinate), glm::vec3(1, 0, 1));
    } 
    
    return result;
}

#endif

