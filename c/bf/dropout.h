#ifndef BF_DROPOUT_H_
#define BF_DROPOUT_H_

#include "bf/module.h"
#include "bf/tensor.h"

typedef struct dropout_t {
    module_type_id_t type;
    float rate;
    tensor_t* mask;
    tensor_t* x;
} dropout_t;

void dropout_init(module_t* module, float rate);
module_t* dropout(float rate);
void dropout_free(module_t* module);
void dropout_zero_grad(module_t* module);
tensor_t* dropout_forward(module_t* module, tensor_t* x, bool is_t);
tensor_t* dropout_backward(module_t* module, tensor_t* dy);
void dropout_update_step(module_t* module, float lr);

#endif  /* BF_DROPOUT_H_ */
