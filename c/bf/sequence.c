#include "sequence.h"

#include <stdlib.h>

void sequence_init(module_t* module, int num_layers, module_t** layers) {
    sequence_t* f = (sequence_t*)module;
    f->type = SEQUENCE;
    f->num_layers = num_layers;
    f->layers = layers;
}

module_t* pp_sequence(int num_layers, ...) {
    va_list args;
    va_start(args, num_layers);
    module_t** layers = (module_t**)malloc(num_layers * sizeof(module_t*));
    for (int i = 0; i < num_layers; ++i) {
        layers[i] = va_arg(args, module_t*);
    }
    va_end(args);
    module_t* f = (module_t*)malloc(sizeof(sequence_t));
    sequence_init(f, num_layers, layers);
    return f;
}

void sequence_free(module_t* module) {
    sequence_t* f = (sequence_t*)module;
    for (int i = 0; i < f->num_layers; ++i) {
        module_t* sub_module = f->layers[i];
        module_free(sub_module);
        free(sub_module);
    }
    free(f->layers);
}

void sequence_zero_grad(module_t* module) {
    sequence_t* f = (sequence_t*)module;
    for (int i = 0; i < f->num_layers; ++i) {
        module_t* sub_module = f->layers[i];
        module_zero_grad(sub_module);
    }
}

tensor_t* sequence_forward(module_t* module, tensor_t* x, bool is_t) {
    sequence_t* f = (sequence_t*)module;
    for (int i = 0; i < f->num_layers; ++i) {
        module_t* sub_module = f->layers[i];
        tensor_t* y = module_forward(sub_module, x, is_t);
        if (i) {
            tensor_free(x);
            free(x);
        }
        x = y;
    }
    return x;
}

tensor_t* sequence_backward(module_t* module, tensor_t* dy) {
    sequence_t* f = (sequence_t*)module;
    for (int i = 0; i < f->num_layers; ++i) {
        module_t* sub_module = f->layers[f->num_layers - i - 1];
        tensor_t* dx = module_backward(sub_module, dy);
        if (i) {
            tensor_free(dy);
            free(dy);
        }
        dy = dx;
    }
    return dy;
}

void sequence_update_step(module_t* module, float lr) {
    sequence_t* f = (sequence_t*)module;
    for (int i = 0; i < f->num_layers; ++i) {
        module_t* sub_module = f->layers[i];
        module_update_step(sub_module, lr);
    }
}
