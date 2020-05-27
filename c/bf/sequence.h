#ifndef BF_SEQUENCE_H_
#define BF_SEQUENCE_H_

#include "bf/module.h"
#include "bf/pp_narg.h"
#include "bf/tensor.h"

typedef struct sequence_t {
    module_type_id_t type;
    int num_layers;
    module_t** layers;
} sequence_t;

void sequence_init(module_t* module, int num_layers, module_t** layers);
module_t* pp_sequence(int num_layers, ...);
#define sequence(...) pp_sequence(PP_NARG(__VA_ARGS__), __VA_ARGS__)
void sequence_free(module_t* module);
void sequence_zero_grad(module_t* module);
tensor_t* sequence_forward(module_t* module, tensor_t* x, bool is_t);
tensor_t* sequence_backward(module_t* module, tensor_t* dy);
void sequence_update_step(module_t* module, float lr);

#endif  /* BF_SEQUENCE_H_ */
