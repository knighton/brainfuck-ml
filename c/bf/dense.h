#ifndef BF_DENSE_H_
#define BF_DENSE_H_

#include "bf/module.h"
#include "bf/tensor.h"

typedef struct dense_t {
    module_type_id_t type;
    int x_dim;
    int y_dim;
    tensor_t* weight;
    tensor_t* dweight;
    tensor_t* bias;
    tensor_t* dbias;
    tensor_t* x;
} dense_t;

void dense_init(module_t* module, int x_dim, int y_dim);
module_t* dense(int x_dim, int y_dim);
void dense_free(module_t* module);
void dense_zero_grad(module_t* module);
tensor_t* dense_forward(module_t* module, tensor_t* x, bool is_t);
tensor_t* dense_backward(module_t* module, tensor_t* dy);
void dense_update_step(module_t* module, float lr);

#endif  /* BF_DENSE_H_ */
