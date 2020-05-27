#include "tensor.h"

#include <assert.h>
#include <stdlib.h>

#include "bf/prng.h"

void tensor_init_common(tensor_t* x, int num_args, va_list* args) {
    assert(num_args <= NDIM_MAX);
    x->size = 1;
    x->ndim = num_args;
    for (int i = 0; i < num_args; ++i) {
        int arg = va_arg(*args, int);
        x->size *= arg;
        x->shape[i] = arg;
    }
    x->data = (float*)malloc(x->size * sizeof(float));
}

void pp_tensor_init(tensor_t* x, int num_args, ...) {
    va_list args;
    va_start(args, num_args);
    tensor_init_common(x, num_args, &args);
    va_end(args);
}

tensor_t* empty(void) {
    tensor_t* x = (tensor_t*)malloc(sizeof(tensor_t));
    x->size = 0;
    x->ndim = 0;
    x->data = NULL;
    return x;
}

tensor_t* pp_tensor(int num_args, ...) {
    tensor_t* x = (tensor_t*)malloc(sizeof(tensor_t));
    va_list args;
    va_start(args, num_args);
    tensor_init_common(x, num_args, &args);
    va_end(args);
    return x;
}

void tensor_free(tensor_t* x) {
    if (x->data) {
        free(x->data);
    }
}

void tensor_fill(tensor_t* x, float a) {
    for (int i = 0; i < x->size; ++i) {
        x->data[i] = a;
    }
}

tensor_t* pp_full(float a, int num_args, ...) {
    tensor_t* x = (tensor_t*)malloc(sizeof(tensor_t));
    va_list args;
    va_start(args, num_args);
    tensor_init_common(x, num_args, &args);
    va_end(args);
    tensor_fill(x, a);
    return x;
}

void tensor_arange(tensor_t* x) {
    for (int i = 0; i < x->size; ++i) {
        x->data[i] = i;
    }
}

tensor_t* pp_arange(int num_args, ...) {
    tensor_t* x = (tensor_t*)malloc(sizeof(tensor_t));
    va_list args;
    va_start(args, num_args);
    tensor_init_common(x, num_args, &args);
    va_end(args);
    tensor_arange(x);
    return x;
}

void tensor_uniform(tensor_t* x, float low, float high) {
    for (int i = 0; i < x->size; ++i) {
        x->data[i] = prng_rand(low, high);
    }
}

tensor_t* pp_uniform(float low, float high, int num_args, ...) {
    tensor_t* x = (tensor_t*)malloc(sizeof(tensor_t));
    va_list args;
    va_start(args, num_args);
    tensor_init_common(x, num_args, &args);
    va_end(args);
    tensor_uniform(x, low, high);
    return x;
}

void tensor_set(tensor_t* x, tensor_t* a) {
    if (x->size != a->size) {
        x->size = a->size;
        if (x->data) {
            free(x->data);
        }
        x->data = (float*)malloc(a->size * sizeof(float));
    }
    x->ndim = a->ndim;
    for (int i = 0; i < a->ndim; ++i) {
        x->shape[i] = a->shape[i];
    }
    for (int i = 0; i < a->size; ++i) {
        x->data[i] = a->data[i];
    }
}

tensor_t* tensor_clone(tensor_t* a) {
    tensor_t* x = empty();
    tensor_set(x, a);
    return x;
}

void tensor_update_step(tensor_t* x, tensor_t* dx, float lr) {
    assert(x->size == dx->size);
    for (int i = 0; i < x->size; ++i) {
        x->data[i] -= lr * dx->data[i];
    }
}
