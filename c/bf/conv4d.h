#ifndef BF_CONV4D_H_
#define BF_CONV4D_H_

#include "bf/coord.h"
#include "bf/module.h"
#include "bf/tensor.h"

typedef struct conv4d_t {
    module_type_id_t type;
    int x_channels;
    int y_channels;
    coord4d_t face;
    coord4d_t stride;
    coord4d_t pad;
    tensor_t* weight;
    tensor_t* dweight;
    tensor_t* bias;
    tensor_t* dbias;
    tensor_t* x;
} conv4d_t;

void conv4d_init(module_t* module, int x_channels, int y_channels, int face,
                 int stride, int pad);
module_t* conv4d(int x_channels, int y_channels, int face, int stride,
                 int pad);
void conv4d_free(module_t* module);
void conv4d_zero_grad(module_t* module);
tensor_t* conv4d_forward(module_t* module, tensor_t* x, bool is_t);
tensor_t* conv4d_backward(module_t* module, tensor_t* dy);
void conv4d_update_step(module_t* module, float lr);

#endif  /* BF_CONV4D_H_ */
