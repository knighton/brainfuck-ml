import torch

from .bf import *


def main():
    num_epochs = 20
    batches_per_epoch = 20
    batch_size = 20
    in_dim = 100
    mid_dim = 50
    out_dim = 10
    lr = 0.007

    x = rand(0, 1, (batch_size, in_dim))
    y_true = randint(0, out_dim, (batch_size,))

    model = Sequence(
        Reshape(-1, 1, 2, 2),
        Flatten(),
        Dense(in_dim, mid_dim),
        ReLU(),
        Dropout(0.5),
        Dense(mid_dim, out_dim),
        Softmax(),
    )

    indices = torch.arange(batch_size)
    samples_per_epoch = batches_per_epoch * batch_size

    for epoch_id in range(num_epochs):
        acc = 0
        for batch_id in range(batches_per_epoch):
            model.zero_grad()
            y_pred = model(x, True)
            loss = -y_pred[indices, y_true].log().sum()
            loss.backward()
            model.update_step(lr)
            acc += int((y_pred.argmax(1) == y_true).sum())
        acc = 100.0 * acc / samples_per_epoch
        print('%6d %6.2f' % (epoch_id, acc))


if __name__ == '__main__':
    main()
