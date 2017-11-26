FLAGS = -msse -O3 -Wall -Wextra -DGLM_FORCE_RADIANS=1
CC = cc $(FLAGS) -std=c99 -S
CPP = c++ $(FLAGS) -std=c++14 -c
EMCC = emsdk_portable/emscripten/1.37.22/emcc # Set this yourself to your emcc
LINKER = c++
EMCC_EXPORTS = EXPORTED_FUNCTIONS="['_get_bead_count', '_get_max_beads', '_get_max_lamps', '_add_bead', '_set_lamp', '_reset_cube', '_jolt', '_tick', '_get_camera_center', '_get_cube_element_count', '_get_cube_vertex_count', '_get_cube_vertex_stride', '_get_cube_normal_offset', '_get_bead_vertex_stride', '_get_bead_color_offset', '_get_cube_elements', '_update_cube_vertices', '_update_bead_vertices', '_get_debug_bead_count', '_update_debug_bead_vertices']"

JellyMcJelloFace: client.o jellymjf.s
	$(LINKER) client.o jellymjf.s -o JellyMcJelloFace -lGL -lSDL2
	
client.o: client.cc vec3.h vec3-fallback.inc jellymjf.h beadface.h opengl-functions.inc
	$(CPP) client.cc -o client.o
	
jellymjf.s: jellymjf.c vec3.h vec3-fallback.inc jellymjf.h
	$(CC) jellymjf.c -o jellymjf.s
	
libjellymjf.js: jellymjf.c vec3.h jellymjf.h vec3-fallback.inc
	$(EMCC) jellymjf.c -O3 -s $(EMCC_EXPORTS) -o libjellymjf.js

libjellymjf-wasm.js: jellymjf.c jellymjf.h
	$(EMCC) jellymjf.c -O3 -s WASM=1 -s $(EMCC_EXPORTS) -o libjellymjf-wasm.js
	

