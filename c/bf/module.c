#include "module.h"

#include <stddef.h>

#include "bf/debug.h"
#include "bf/dense.h"
#include "bf/dropout.h"
#include "bf/relu.h"
#include "bf/sequence.h"
#include "bf/softmax.h"

typedef void (*module_free_t)(module_t* module);
typedef void (*module_zero_grad_t)(module_t* module);
typedef tensor_t* (*module_forward_t)(module_t* module, tensor_t* x,
                                      bool is_t);
typedef tensor_t* (*module_backward_t)(module_t* module, tensor_t* dy);
typedef void (*module_update_step_t)(module_t* module, float lr);

typedef struct module_api_t {
    module_free_t free;
    module_zero_grad_t zero_grad;
    module_forward_t forward;
    module_backward_t backward;
    module_update_step_t update_step;
} module_api_t;

#define TODO(module) { NULL, NULL, NULL, NULL, NULL }

#define API(module) { \
    module##_free, \
    module##_zero_grad, \
    module##_forward, \
    module##_backward, \
    module##_update_step \
}

static module_api_t APIS[] = {
    API(debug),
    API(dense),
    API(dropout),
    API(relu),
    API(sequence),
    API(softmax),
};

#undef TODO

void module_free(module_t* module) {
    APIS[module->type].free(module);
}

void module_zero_grad(module_t* module) {
    APIS[module->type].zero_grad(module);
}

tensor_t* module_forward(module_t* module, tensor_t* x, bool is_t) {
    return APIS[module->type].forward(module, x, is_t);
}

tensor_t* module_backward(module_t* module, tensor_t* dy) {
    return APIS[module->type].backward(module, dy);
}

void module_update_step(module_t* module, float lr) {
    APIS[module->type].update_step(module, lr);
}
