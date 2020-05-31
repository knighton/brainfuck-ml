#include <stdio.h>
#include <stdlib.h>

#include "bf/bf.h"

tensor_t* get_loss(tensor_t* y_pred, int* y_true) {
    int batch_size = y_pred->shape[0];
    int dim = y_pred->shape[1];
    tensor_t* dy = tensor_clone(y_pred);
    for (int i = 0; i < batch_size; ++i) {
        --dy->data[i * dim + y_true[i]];
    }
    return dy;
}

int get_acc(tensor_t* y_pred, int* y_true) {
    int batch_size = y_pred->shape[0];
    int dim = y_pred->shape[1];
    int acc = 0;
    for (int i = 0; i < batch_size; ++i) {
        int max_cls = 0;
        float max_val = y_pred->data[i * dim];
        for (int j = 1; j < dim; ++j) {
            float val = y_pred->data[i * dim + j];
            if (max_val < val) {
                max_cls = j;
                max_val = val;
            }
        }
        if (max_cls == y_true[i]) {
            ++acc;
        }
    }
    return acc;
}

int main() {
    int num_epochs = 20;
    int batches_per_epoch = 20;
    int batch_size = 20;
    int in_dim = 100;
    int mid_dim = 50;
    int out_dim = 10;
    float lr = 0.007;

    tensor_t* x = uniform(0, 1, batch_size, in_dim);
    int* y_true = (int*)malloc(batch_size * sizeof(int));
    for (int i = 0; i < batch_size; ++i) {
        y_true[i] = prng_randint(0, out_dim);
    }

    module_t* model = sequence(
        reshape(-1, 1, 2, 2),
        flatten(),
        dense(in_dim, mid_dim),
        batchnorm0d(mid_dim, 0.9, 1e-5),
        relu(),
        dropout(0.5),
        dense(mid_dim, out_dim),
        softmax()
    );

    float acc_to_pct = 100.0 / (batches_per_epoch * batch_size);
    for (int i = 0; i < num_epochs; ++i) {
        int acc = 0;
        for (int j = 0; j < batches_per_epoch; ++j) {
            module_zero_grad(model);
            tensor_t* y_pred = module_forward(model, x, true);
            tensor_t* dy = get_loss(y_pred, y_true);
            tensor_t* dx = module_backward(model, dy);
            module_update_step(model, lr);
            acc += get_acc(y_pred, y_true);

            tensor_free(dx);
            free(dx);
            tensor_free(dy);
            free(dy);
            tensor_free(y_pred);
            free(y_pred);
        }
        float pct = acc * acc_to_pct;
        printf("%6d %6.2f\n", i, pct);
    }

    module_free(model);
    free(model);

    tensor_free(x);
    free(x);
    free(y_true);
}
