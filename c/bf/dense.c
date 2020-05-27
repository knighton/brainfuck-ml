#include "dense.h"

#include <assert.h>
#include <stdlib.h>

void dense_init(module_t* module, int x_dim, int y_dim) {
    assert(0 < x_dim);
    assert(0 < y_dim);
    dense_t* f = (dense_t*)module;
    f->type = DENSE;
    f->x_dim = x_dim;
    f->y_dim = y_dim;
    f->weight = uniform(-0.05, 0.05, x_dim, y_dim);
    f->dweight = zeros(x_dim, y_dim);
    f->bias = uniform(-0.05, 0.05, y_dim);
    f->dbias = zeros(y_dim);
    f->x = empty();
}

module_t* dense(int x_dim, int y_dim) {
    module_t* module = (module_t*)malloc(sizeof(dense_t));
    dense_init(module, x_dim, y_dim);
    return module;
}

void dense_free(module_t* module) {
    dense_t* f = (dense_t*)module;
    tensor_free(f->weight);
    tensor_free(f->dweight);
    tensor_free(f->bias);
    tensor_free(f->dbias);
    tensor_free(f->x);
    free(f->weight);
    free(f->dweight);
    free(f->bias);
    free(f->dbias);
    free(f->x);
}

void dense_zero_grad(module_t* module) {
    dense_t* f = (dense_t*)module;
    tensor_zero(f->dweight);
    tensor_zero(f->dbias);
}

tensor_t* dense_forward(module_t* module, tensor_t* x, bool is_t) {
    dense_t* f = (dense_t*)module;
    if (is_t) {
        tensor_set(f->x, x);
    }
    int count = x->shape[0];
    assert(f->x_dim == x->shape[1]);
    tensor_t* y = zeros(count, f->y_dim);
    for (int n = 0; n < count; ++n) {
        for (int yd = 0; yd < f->y_dim; ++yd) {
            int y_idx = n * f->y_dim + yd;
            float y_val = 0;
            for (int xd = 0; xd < f->x_dim; ++xd) {
                int x_idx = n * f->x_dim + xd;
                int w_idx = xd * f->y_dim + yd;
                y_val += x->data[x_idx] * f->weight->data[w_idx];
            }
            y_val += f->bias->data[yd];
            y->data[y_idx] = y_val;
        }
    }
    return y;
}

tensor_t* dense_backward(module_t* module, tensor_t* dy) {
    dense_t* f = (dense_t*)module;
    int count = dy->shape[0];
    assert(f->y_dim == dy->shape[1]);
    tensor_t* dx = zeros(count, f->x_dim);
    for (int n = 0; n < count; ++n) {
        for (int yd = 0; yd < f->y_dim; ++yd) {
            int y_idx = n * f->y_dim + yd;
            float dy_val = dy->data[y_idx];
            for (int xd = 0; xd < f->x_dim; ++xd) {
                int x_idx = n * f->x_dim + xd;
                int w_idx = xd * f->y_dim + yd;
                f->dweight->data[w_idx] += dy_val * f->x->data[x_idx];
                dx->data[x_idx] += dy_val * f->weight->data[w_idx];
            }
            f->dbias->data[yd] += dy_val;
        }
    }
    return dx;
}

void dense_update_step(module_t* module, float lr) {
    dense_t* f = (dense_t*)module;
    tensor_update_step(f->weight, f->dweight, lr);
    tensor_update_step(f->bias, f->dbias, lr);
}
