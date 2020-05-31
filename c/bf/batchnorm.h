#ifndef BF_BATCHNORM_H_
#define BF_BATCHNORM_H_

#include "bf/module.h"
#include "bf/tensor.h"

typedef struct batchnorm_t {
    module_type_id_t type;
    int dim;
    float momentum;
    float eps;
    int ndim;
    tensor_t* gamma;
    tensor_t* dgamma;
    tensor_t* beta;
    tensor_t* dbeta;
    tensor_t* mov_mean;
    tensor_t* mov_std;
    tensor_t* x;
} batchnorm_t;

void batchnorm_init(module_t* module, int dim, float momentum, float eps,
                    int ndim);
module_t* batchnorm(int dim, float momentum, float eps, int ndim);
#define batchnorm0d(...) batchnorm(__VA_ARGS__, 0)
#define batchnorm1d(...) batchnorm(__VA_ARGS__, 1)
#define batchnorm2d(...) batchnorm(__VA_ARGS__, 2)
#define batchnorm3d(...) batchnorm(__VA_ARGS__, 3)
#define batchnorm4d(...) batchnorm(__VA_ARGS__, 4)
void batchnorm_free(module_t* module);
void batchnorm_zero_grad(module_t* module);
tensor_t* batchnorm_forward(module_t* module, tensor_t* x, bool is_t);
tensor_t* batchnorm_backward(module_t* module, tensor_t* dy);
void batchnorm_update_step(module_t* module, float lr);

#endif  /* BF_BATCHNORM_H_ */
