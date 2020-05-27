#include "relu.h"

#include <stdlib.h>

#include "bf/lint.h"

void relu_init(module_t* module) {
    relu_t* f = (relu_t*)module;
    f->type = RELU;
    f->x = empty();
}

module_t* relu(void) {
    module_t* module = (module_t*)malloc(sizeof(relu_t));
    relu_init(module);
    return module;
}

void relu_free(module_t* module) {
    relu_t* f = (relu_t*)module;
    tensor_free(f->x);
    free(f->x);
}

void relu_zero_grad(module_t* module) {
    UNUSED(module);
}

tensor_t* relu_forward(module_t* module, tensor_t* x, bool is_t) {
    relu_t* f = (relu_t*)module;
    if (is_t) {
        tensor_set(f->x, x);
    }
    tensor_t* y = tensor_clone(x);
    for (int i = 0; i < y->size; ++i) {
        if (y->data[i] < 0) {
            y->data[i] = 0;
        }
    }
    return y;
}

tensor_t* relu_backward(module_t* module, tensor_t* dy) {
    relu_t* f = (relu_t*)module;
    tensor_t* dx = tensor_clone(dy);
    for (int i = 0; i < f->x->size; ++i) {
        if (f->x->data[i] < 0) {
            dx->data[i] = 0;
        }
    }
    return dx;
}

void relu_update_step(module_t* module, float lr) {
    UNUSED(module);
    UNUSED(lr);
}
