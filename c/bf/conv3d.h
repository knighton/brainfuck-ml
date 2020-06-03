#ifndef BF_CONV3D_H_
#define BF_CONV3D_H_

#include "bf/coord.h"
#include "bf/module.h"
#include "bf/tensor.h"

typedef struct conv3d_t {
    module_type_id_t type;
    int x_channels;
    int y_channels;
    coord3d_t face;
    coord3d_t stride;
    coord3d_t pad;
    tensor_t* weight;
    tensor_t* dweight;
    tensor_t* bias;
    tensor_t* dbias;
    tensor_t* x;
} conv3d_t;

void conv3d_init(module_t* module, int x_channels, int y_channels, int face,
                 int stride, int pad);
module_t* conv3d(int x_channels, int y_channels, int face, int stride,
                 int pad);
void conv3d_free(module_t* module);
void conv3d_zero_grad(module_t* module);
tensor_t* conv3d_forward(module_t* module, tensor_t* x, bool is_t);
tensor_t* conv3d_backward(module_t* module, tensor_t* dy);
void conv3d_update_step(module_t* module, float lr);

#endif  /* BF_CONV3D_H_ */
