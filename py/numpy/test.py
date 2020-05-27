import numpy as np

from .bf import *


def main():
    num_epochs = 10
    batches_per_epoch = 10
    batch_size = 10
    in_dim = 100
    mid_dim = 50
    out_dim = 10
    lr = 0.007

    x = rand(0, 1, (batch_size, in_dim))
    y_true = randint(0, out_dim, (batch_size,))

    model = Sequence(
        Dense(in_dim, mid_dim),
        ReLU(),
        Dense(mid_dim, out_dim),
        Softmax(),
    )

    indices = np.arange(batch_size)
    samples_per_epoch = batches_per_epoch * batch_size

    for epoch_id in range(num_epochs):
        acc = 0
        for batch_id in range(batches_per_epoch):
            model.zero_grad()
            y_pred = model(x, True)
            dy_pred = y_pred.copy()
            dy_pred[indices, y_true] -= 1
            model.backward(dy_pred)
            model.update_step(lr)
            acc += int((y_pred.argmax(1) == y_true).sum())
        acc = 100.0 * acc / samples_per_epoch
        print('%6d %6.2f' % (batch_id, acc))


if __name__ == '__main__':
    main()
