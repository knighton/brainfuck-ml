#ifndef BF_TENSOR_H_
#define BF_TENSOR_H_

#include <stdarg.h>
#include <stdint.h>

#include "bf/pp_narg.h"

#define NDIM_MAX 6

typedef struct tensor_t {
    int size;
    int ndim;
    int shape[NDIM_MAX];
    float* data;
} tensor_t;

typedef struct shape_t {
    int size;
    int ndim;
    int shape[NDIM_MAX];
} shape_t;

void pp_tensor_init(tensor_t* x, int num_args, ...);
#define tensor_init(x, ...) \
    pp_tensor_init(x, PP_NARG(__VA_ARGS__), __VA_ARGS__)
tensor_t* empty(void);
tensor_t* pp_tensor(int num_args, ...);
#define tensor(...) pp_tensor(PP_NARG(__VA_ARGS__), __VA_ARGS__)
void tensor_free(tensor_t* x);

void tensor_fill(tensor_t* x, float a);
#define tensor_one(x) tensor_fill(x, 1)
#define tensor_zero(x) tensor_fill(x, 0)
tensor_t* pp_full(float a, int num_args, ...);
#define full(a, ...) pp_full(a, PP_NARG(__VA_ARGS__), __VA_ARGS__)
#define ones(...) full(1, __VA_ARGS__)
#define zeros(...) full(0, __VA_ARGS__)

void tensor_arange(tensor_t* x);
tensor_t* pp_arange(int num_args, ...);
#define arange(...) pp_arange(PP_NARG(__VA_ARGS__), __VA_ARGS__)

void tensor_uniform(tensor_t* x, float low, float high);
tensor_t* pp_uniform(float low, float high, int num_args, ...);
#define uniform(low, high, ...) \
    pp_uniform(low, high, PP_NARG(__VA_ARGS__), __VA_ARGS__)

void tensor_set(tensor_t* x, tensor_t* a);
tensor_t* tensor_clone(tensor_t* a);

void tensor_update_step(tensor_t* x, tensor_t* dx, float lr);

void tensor_shape(tensor_t* t, shape_t* s);
tensor_t* tensor_reshape(tensor_t* t, shape_t* s);

#endif  /* BF_TENSOR_H_ */
