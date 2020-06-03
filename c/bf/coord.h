#ifndef BF_COORDS_H_
#define BF_COORDS_H_

#include <stdint.h>

#include "bf/pp_narg.h"

typedef struct coord4d_t {
    int time;
    int depth;
    int height;
    int width;
} coord4d_t;

typedef struct coord3d_t {
    int depth;
    int height;
    int width;
} coord3d_t;

typedef struct coord2d_t {
    int height;
    int width;
} coord2d_t;

typedef struct coord1d_t {
    int width;
} coord1d_t;

int pp_coord(int num_args, ...);
#define coord(...) pp_coord(PP_NARG(__VA_ARGS__), __VA_ARGS__)

void coord4d_init(int x, coord4d_t* coord, int min);
void coord3d_init(int x, coord3d_t* coord, int min);
void coord2d_init(int x, coord2d_t* coord, int min);
void coord1d_init(int x, coord1d_t* coord, int min);

#endif  /* BF_COORDS_H_ */
