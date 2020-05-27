#include "softmax.h"

#include <assert.h>
#include <math.h>
#include <stdlib.h>

#include "bf/lint.h"

void softmax_init(module_t* module) {
    softmax_t* f = (softmax_t*)module;
    f->type = SOFTMAX;
}

module_t* softmax(void) {
    module_t* module = (module_t*)malloc(sizeof(softmax_t));
    softmax_init(module);
    return module;
}

void softmax_free(module_t* module) {
    UNUSED(module);
}

void softmax_zero_grad(module_t* module) {
    UNUSED(module);
}

tensor_t* softmax_forward(module_t* module, tensor_t* x, bool is_t) {
    UNUSED(module);
    assert(x->ndim == 2);
    int batch_size = x->shape[0];
    int dim = x->shape[1];
    tensor_t* y = zeros(batch_size, dim);
    for (int n = 0; n < batch_size; ++n) {
        float max = x->data[n * dim];
        for (int d = 1; d < dim; ++d) {
            int key = n * dim + d;
            float val = x->data[key];
            if (max < val) {
                max = val;
            }
        }
        float sum = 0;
        for (int d = 0; d < dim; ++d) {
            int key = n * dim + d;
            float val = exp(x->data[key] - max);
            y->data[key] = val;
            sum += val;
        }
        for (int d = 0; d < dim; ++d) {
            int key = n * dim + d;
            y->data[key] /= sum;
        }
    }
    return y;
}

tensor_t* softmax_backward(module_t* module, tensor_t* dy) {
    UNUSED(module);
    return tensor_clone(dy);
}

void softmax_update_step(module_t* module, float lr) {
    UNUSED(module);
    UNUSED(lr);
}
