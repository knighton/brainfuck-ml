#ifndef BF_SOFTMAX_H_
#define BF_SOFTMAX_H_

#include "bf/module.h"
#include "bf/tensor.h"

typedef struct softmax_t {
    module_type_id_t type;
} softmax_t;

void softmax_init(module_t* module);
module_t* softmax(void);
void softmax_free(module_t* module);
void softmax_zero_grad(module_t* module);
tensor_t* softmax_forward(module_t* module, tensor_t* x, bool is_t);
tensor_t* softmax_backward(module_t* module, tensor_t* dy);
void softmax_update_step(module_t* module, float lr);

#endif  /* BF_SOFTMAX_H_ */
