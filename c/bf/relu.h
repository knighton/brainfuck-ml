#ifndef BF_RELU_H_
#define BF_RELU_H_

#include "bf/module.h"
#include "bf/tensor.h"

typedef struct relu_t {
    module_type_id_t type;
    tensor_t* x;
} relu_t;

void relu_init(module_t* module);
module_t* relu(void);
void relu_free(module_t* module);
void relu_zero_grad(module_t* module);
tensor_t* relu_forward(module_t* module, tensor_t* x, bool is_t);
tensor_t* relu_backward(module_t* module, tensor_t* dy);
void relu_update_step(module_t* module, float lr);

#endif  /* BF_RELU_H_ */
