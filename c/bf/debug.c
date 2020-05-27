#include "bf/debug.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "bf/lint.h"

void debug_init(module_t* module, const char* text) {
    debug_t* f = (debug_t*)module;
    f->type = DEBUG;
    f->text = text;
}

module_t* debug(const char* text) {
    module_t* module = (module_t*)malloc(sizeof(debug_t));
    debug_init(module, text);
    return module;
}

void debug_free(module_t* module) {
    UNUSED(module);
}

void debug_zero_grad(module_t* module) {
    UNUSED(module);
}

void debug_print(const char* text, tensor_t* x, FILE* out) {
    fprintf(out, "%s %d @ (", text, x->size);
    if (x->ndim) {
        fprintf(out, "%d", x->shape[0]);
    }
    for (int i = 1; i < x->ndim; ++i) {
        fprintf(out, ", %d", x->shape[i]);
    }
    fprintf(out, ")\n");
    fflush(out);
}

tensor_t* debug_forward(module_t* module, tensor_t* x, bool is_t) {
    UNUSED(is_t);
    debug_t* f = (debug_t*)module;
    debug_print(f->text, x, stdout);
    return tensor_clone(x);
}

tensor_t* debug_backward(module_t* module, tensor_t* dy) {
    UNUSED(module);
    return tensor_clone(dy);
}

void debug_update_step(module_t* module, float lr) {
    UNUSED(module);
    UNUSED(lr);
}
