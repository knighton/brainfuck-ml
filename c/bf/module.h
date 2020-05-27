#ifndef BF_MODULE_H_
#define BF_MODULE_H_

#include <stdbool.h>

#include "bf/tensor.h"

typedef enum module_type_id_t {
    DENSE,
    DROPOUT,
    RELU,
    SEQUENCE,
    SOFTMAX,
} module_type_id_t;

typedef struct module_t {
    module_type_id_t type;
} module_t;

void module_free(module_t* module);
void module_zero_grad(module_t* module);
tensor_t* module_forward(module_t* module, tensor_t* x, bool is_t);
tensor_t* module_backward(module_t* module, tensor_t* dy);
void module_update_step(module_t* module, float lr);

#endif  /* BF_MODULE_H_ */
