#include "reshape.h"

#include <assert.h>
#include <stdlib.h>

#include <stdio.h>

#include "bf/lint.h"

void reshape_init_common(module_t* module, int num_args, va_list* args) {
    reshape_t* f = (reshape_t*)module;
    f->type = RESHAPE;
    assert(num_args + 1 <= NDIM_MAX);
    f->y.size = 1;
    f->y.ndim = num_args + 1;
    f->y.shape[0] = 0;
    bool has_wildcard = false;
    for (int i = 0; i < num_args; ++i) {
        int arg = va_arg(*args, int);
        if (has_wildcard) {
            assert(1 <= arg);
        } else {
            if (arg == -1) {
                has_wildcard = true;
            } else {
                assert(1 <= arg);
            }
        }
        f->y.size *= arg;
        f->y.shape[i + 1] = arg;
    }
    f->x.size = 0;
    f->x.ndim = 0;
}

void pp_reshape_init(module_t* module, int num_args, ...) {
    va_list args;
    va_start(args, num_args);
    reshape_init_common(module, num_args, &args);
    va_end(args);
}

module_t* pp_reshape(int num_args, ...) {
    module_t* module = (module_t*)malloc(sizeof(reshape_t));
    va_list args;
    va_start(args, num_args);
    reshape_init_common(module, num_args, &args);
    va_end(args);
    return module;
}

void reshape_free(module_t* module) {
    UNUSED(module);
}

void reshape_zero_grad(module_t* module) {
    UNUSED(module);
}

tensor_t* reshape_forward(module_t* module, tensor_t* x, bool is_t) {
    UNUSED(is_t);
    reshape_t* f = (reshape_t*)module;
    tensor_shape(x, &f->x);
    int batch_size = x->shape[0];
    f->y.size *= batch_size;
    f->y.shape[0] = batch_size;
    tensor_t* y = tensor_reshape(x, &f->y);
    f->y.size /= batch_size;
    return y;
}

tensor_t* reshape_backward(module_t* module, tensor_t* dy) {
    reshape_t* f = (reshape_t*)module;
    return tensor_reshape(dy, &f->x);
}

void reshape_update_step(module_t* module, float lr) {
    UNUSED(module);
    UNUSED(lr);
}
