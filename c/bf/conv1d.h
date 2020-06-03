#ifndef BF_CONV1D_H_
#define BF_CONV1D_H_

#include "bf/coord.h"
#include "bf/module.h"
#include "bf/tensor.h"

typedef struct conv1d_t {
    module_type_id_t type;
    int x_channels;
    int y_channels;
    coord1d_t face;
    coord1d_t stride;
    coord1d_t pad;
    tensor_t* weight;
    tensor_t* dweight;
    tensor_t* bias;
    tensor_t* dbias;
    tensor_t* x;
} conv1d_t;

void conv1d_init(module_t* module, int x_channels, int y_channels, int face,
                 int stride, int pad);
module_t* conv1d(int x_channels, int y_channels, int face, int stride,
                 int pad);
void conv1d_free(module_t* module);
void conv1d_zero_grad(module_t* module);
tensor_t* conv1d_forward(module_t* module, tensor_t* x, bool is_t);
tensor_t* conv1d_backward(module_t* module, tensor_t* dy);
void conv1d_update_step(module_t* module, float lr);

#endif  /* BF_CONV1D_H_ */
