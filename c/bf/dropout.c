#include "dropout.h"

#include <assert.h>
#include <stdlib.h>

#include <stdio.h>

#include "bf/lint.h"
#include "bf/prng.h"

void dropout_init(module_t* module, float rate) {
    assert(0 <= rate);
    assert(rate <= 1);
    dropout_t* f = (dropout_t*)module;
    f->type = DROPOUT;
    f->rate = rate;
    f->mask = empty();
}

module_t* dropout(float rate) {
    module_t* module = (module_t*)malloc(sizeof(dropout_t));
    dropout_init(module, rate);
    return module;
}

void dropout_free(module_t* module) {
    dropout_t* f = (dropout_t*)module;
    tensor_free(f->mask);
    free(f->mask);
}

void dropout_zero_grad(module_t* module) {
    UNUSED(module);
}

tensor_t* dropout_forward(module_t* module, tensor_t* x, bool is_t) {
    dropout_t* f = (dropout_t*)module;
    tensor_t* y = tensor_clone(x);
    if (is_t) {
        tensor_set(f->mask, x);
        for (int i = 0; i < y->size; ++i) {
            if (f->rate <= prng_rand(0, 1)) {
                f->mask->data[i] = 1;
                y->data[i] /= f->rate;
            } else {
                f->mask->data[i] = 0;
                y->data[i] = 0;
            }
        }
    }
    return y;
}

tensor_t* dropout_backward(module_t* module, tensor_t* dy) {
    dropout_t* f = (dropout_t*)module;
    tensor_t* dx = tensor_clone(dy);
    for (int i = 0; i < dx->size; ++i) {
        dx->data[i] *= f->mask->data[i];
    }
    return dx;
}

void dropout_update_step(module_t* module, float lr) {
    UNUSED(module);
    UNUSED(lr);
}
