#ifndef BF_RESHAPE_H_
#define BF_RESHAPE_H_

#include "bf/module.h"
#include "bf/tensor.h"

typedef struct reshape_t {
    module_type_id_t type;
    shape_t y;
    shape_t x;
} reshape_t;

void pp_reshape_init(module_t* module, int num_args, ...);
#define reshape_init(module, ...) \
    pp_reshape_init(module, PP_NARG(__VA_ARGS__), __VA_ARGS__)
module_t* pp_reshape(int num_args, ...);
#define reshape(...) pp_reshape(PP_NARG(__VA_ARGS__), __VA_ARGS__)
#define flatten() reshape(-1)
void reshape_free(module_t* module);
void reshape_zero_grad(module_t* module);
tensor_t* reshape_forward(module_t* module, tensor_t* x, bool is_t);
tensor_t* reshape_backward(module_t* module, tensor_t* dy);
void reshape_update_step(module_t* module, float lr);

#endif  /* BF_RESHAPE_H_ */
