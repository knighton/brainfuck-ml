#ifndef BF_DEBUG_H_
#define BF_DEBUG_H_

#include "bf/module.h"
#include "bf/tensor.h"

typedef struct debug_t {
    module_type_id_t type;
    const char* text;
} debug_t;

void debug_init(module_t* module, const char* text);
module_t* debug(const char* text);
void debug_free(module_t* module);
void debug_zero_grad(module_t* module);
tensor_t* debug_forward(module_t* module, tensor_t* x, bool is_t);
tensor_t* debug_backward(module_t* module, tensor_t* dy);
void debug_update_step(module_t* module, float lr);

#endif  /* BF_DEBUG_H_ */
