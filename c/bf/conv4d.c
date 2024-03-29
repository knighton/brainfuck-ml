#include "bf/conv4d.h"

#include <assert.h>
#include <stdlib.h>

void conv4d_init(module_t* module, int x_channels, int y_channels, int face,
                 int stride, int pad) {
    conv4d_t* f = (conv4d_t*)module;
    f->type = CONV4D;
    f->x_channels = x_channels;
    f->y_channels = y_channels;
    coord4d_init(face, &f->face, 1);
    coord4d_init(stride, &f->stride, 1);
    coord4d_init(pad, &f->pad, 0);
    f->weight = uniform(-0.05, 0.05, x_channels, y_channels, f->face.time,
                        f->face.depth, f->face.height, f->face.width);
    f->dweight = zeros(x_channels, y_channels, f->face.time, f->face.depth,
                       f->face.height, f->face.width);
    f->bias = uniform(-0.05, 0.05, y_channels);
    f->dbias = zeros(y_channels);
    f->x = empty();
}

module_t* conv4d(int x_channels, int y_channels, int face, int stride,
                 int pad) {
    module_t* module = (module_t*)malloc(sizeof(conv4d_t));
    conv4d_init(module, x_channels, y_channels, face, stride, pad);
    return module;
}

void conv4d_free(module_t* module) {
    conv4d_t* f = (conv4d_t*)module;
    tensor_free(f->weight);
    tensor_free(f->dweight);
    tensor_free(f->bias);
    tensor_free(f->dbias);
    tensor_free(f->x);
    free(f->weight);
    free(f->dweight);
    free(f->bias);
    free(f->dbias);
    free(f->x);
}

void conv4d_zero_grad(module_t* module) {
    conv4d_t* f = (conv4d_t*)module;
    tensor_zero(f->dweight);
    tensor_zero(f->dbias);
}

tensor_t* conv4d_forward(module_t* module, tensor_t* x, bool is_t) {
    conv4d_t* f = (conv4d_t*)module;
    if (is_t) {
        tensor_set(f->x, x);
    }

    int batch_size = x->shape[0];
    assert(f->x_channels == x->shape[1]);
    int x_time = x->shape[2];
    int x_depth = x->shape[3];
    int x_height = x->shape[4];
    int x_width = x->shape[5];
    int y_time = (x_time + 2 * f->pad.time - f->face.time) /
        f->stride.time + 1;
    int y_depth = (x_depth + 2 * f->pad.depth - f->face.depth) /
        f->stride.depth + 1;
    int y_height = (x_height + 2 * f->pad.height - f->face.height) /
        f->stride.height + 1;
    int y_width = (x_width + 2 * f->pad.width - f->face.width) /
        f->stride.width + 1;
    tensor_t* y = zeros(batch_size, f->y_channels, y_time, y_depth, y_height,
                        y_width);

    for (int y_t = 0, x_t_offset = -f->pad.time; y_t < y_time;
            ++y_t, x_t_offset += f->stride.time) {
        for (int face_t = 0; face_t < f->face.time; ++face_t) {
            int x_t = x_t_offset + face_t;
            if (x_t < 0 || x_time <= x_t) {
                continue;
            }

    for (int y_d = 0, x_d_offset = -f->pad.depth; y_d < y_depth;
            ++y_d, x_d_offset += f->stride.depth) {
        for (int face_d = 0; face_d < f->face.depth; ++face_d) {
            int x_d = x_d_offset + face_d;
            if (x_d < 0 || x_depth <= x_d) {
                continue;
            }

    for (int y_h = 0, x_h_offset = -f->pad.height; y_h < y_height;
            ++y_h, x_h_offset += f->stride.height) {
        for (int face_h = 0; face_h < f->face.height; ++face_h) {
            int x_h = x_h_offset + face_h;
            if (x_h < 0 || x_height <= x_h) {
                continue;
            }

    for (int y_w = 0, x_w_offset = -f->pad.width; y_w < y_width;
            ++y_w, x_w_offset += f->stride.width) {
        for (int face_w = 0; face_w < f->face.width; ++face_w) {
            int x_w = x_w_offset + face_w;
            if (x_w < 0 || x_width <= x_w) {
                continue;
            }

    for (int n = 0; n < batch_size; ++n) {
        for (int y_c = 0; y_c < f->y_channels; ++y_c) {
            float y_val = 0;
            for (int x_c = 0; x_c < f->x_channels; ++x_c) {
                int x_idx = ((((n * f->x_channels + x_c) * x_time + x_t) *
                    x_depth + x_d) * x_height + x_h) * x_width + x_w;
                int w_idx = ((((x_c * f->y_channels + y_c) * f->face.time +
                    face_t) * f->face.depth + face_d) * f->face.height +
                    face_h) * f->face.width + face_w;
                y_val += x->data[x_idx] * f->weight->data[w_idx];
            }
            int y_idx = ((((n * f->y_channels + y_c) * y_time + y_t) *
                y_depth + y_d) * y_height + y_h) * y_width + y_w;
            y->data[y_idx] += y_val;
        }
    }

        }
    }

        }
    }

        }
    }

        }
    }

    for (int n = 0; n < batch_size; ++n) {
        for (int y_c = 0; y_c < f->y_channels; ++y_c) {
            for (int y_t = 0; y_t < y_time; ++y_t) {
                for (int y_d = 0; y_d < y_depth; ++y_d) {
                    for (int y_h = 0; y_h < y_height; ++y_h) {
                        for (int y_w = 0; y_w < y_width; ++y_w) {
                            int y_idx = ((((n * f->y_channels + y_c) * y_time
                                + y_t) * y_depth + y_d) * y_height + y_h) *
                                y_width + y_w;
                            y->data[y_idx] += f->bias->data[y_c];
                        }
                    }
                }
            }
        }
    }

    return y;
}

tensor_t* conv4d_backward(module_t* module, tensor_t* dy) {
    conv4d_t* f = (conv4d_t*)module;

    int batch_size = dy->shape[0];
    int x_time = f->x->shape[2];
    int x_depth = f->x->shape[3];
    int x_height = f->x->shape[4];
    int x_width = f->x->shape[5];
    assert(f->y_channels == dy->shape[1]);
    int y_time = dy->shape[2];
    int y_depth = dy->shape[3];
    int y_height = dy->shape[4];
    int y_width = dy->shape[5];
    tensor_t* dx = zeros(batch_size, f->x_channels, x_time, x_depth,
                         x_height, x_width);

    for (int y_t = 0, x_t_offset = -f->pad.time; y_t < y_time;
            ++y_t, x_t_offset += f->stride.time) {
        for (int face_t = 0; face_t < f->face.time; ++face_t) {
            int x_t = x_t_offset + face_t;
            if (x_t < 0 || x_time <= x_t) {
                continue;
            }

    for (int y_d = 0, x_d_offset = -f->pad.depth; y_d < y_depth;
            ++y_d, x_d_offset += f->stride.depth) {
        for (int face_d = 0; face_d < f->face.depth; ++face_d) {
            int x_d = x_d_offset + face_d;
            if (x_d < 0 || x_depth <= x_d) {
                continue;
            }

    for (int y_h = 0, x_h_offset = -f->pad.height; y_h < y_height;
            ++y_h, x_h_offset += f->stride.height) {
        for (int face_h = 0; face_h < f->face.height; ++face_h) {
            int x_h = x_h_offset + face_h;
            if (x_h < 0 || x_height <= x_h) {
                continue;
            }

    for (int y_w = 0, x_w_offset = -f->pad.width; y_w < y_width;
            ++y_w, x_w_offset += f->stride.width) {
        for (int face_w = 0; face_w < f->face.width; ++face_w) {
            int x_w = x_w_offset + face_w;
            if (x_w < 0 || x_width <= x_w) {
                continue;
            }

    for (int n = 0; n < batch_size; ++n) {
        for (int y_c = 0; y_c < f->y_channels; ++y_c) {
            int y_idx = ((((n * f->y_channels + y_c) * y_time + y_t) *
                y_depth + y_d) * y_height + y_h) * y_width + y_w;
            float dy_val = dy->data[y_idx];
            for (int x_c = 0; x_c < f->x_channels; ++x_c) {
                int x_idx = ((((n * f->x_channels + x_c) * x_time + x_t) *
                    x_depth + x_d) * x_height + x_h) * x_width + x_w;
                int w_idx = ((((x_c * f->y_channels + y_c) * f->face.time +
                    face_t) * f->face.depth + face_d) * f->face.height +
                    face_h) * f->face.width + face_w;
                f->dweight->data[w_idx] += dy_val * f->x->data[x_idx];
                dx->data[x_idx] += dy_val * f->weight->data[w_idx];
            }
        }
    }

        }
    }

        }
    }

        }
    }

        }
    }

    for (int n = 0; n < batch_size; ++n) {
        for (int y_c = 0; y_c < f->y_channels; ++y_c) {
            for (int y_t = 0; y_t < y_time; ++y_t) {
                for (int y_d = 0; y_d < y_depth; ++y_d) {
                    for (int y_h = 0; y_h < y_height; ++y_h) {
                        for (int y_w = 0; y_w < y_width; ++y_w) {
                            int y_idx = (((n * f->y_channels + y_c) * y_depth
                                + y_d) * y_height + y_h) * y_width + y_w;
                            f->dbias->data[y_c] += dy->data[y_idx];
                        }
                    }
                }
            }
        }
    }

    return dx;
}

void conv4d_update_step(module_t* module, float lr) {
    conv4d_t* f = (conv4d_t*)module;
    tensor_update_step(f->weight, f->dweight, lr);
    tensor_update_step(f->bias, f->dbias, lr);
}
