import os
from wurlitzer import pipes
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
with pipes() as (out, err):
    import tensorflow as tf

from .bf import *


def main():
    num_epochs = 50
    batches_per_epoch = 50
    batch_size = 32
    lr = 0.007

    x = rand(0, 1, (batch_size, 8, 2, 2, 2))
    y_true = randint(0, 16, (batch_size,))

    model = Sequence(
        Conv3d(8, 8, 3, 1, 1),
        BatchNorm3d(8),

        Reshape(16, 2, 2),

        ReLU(),
        Conv2d(16, 16, 3, 1, 1),
        BatchNorm2d(16),

        Reshape(32, 2),

        ReLU(),
        Conv1d(32, 32, 3, 1, 1),
        BatchNorm1d(32),

        Flatten(),

        ReLU(),
        Dropout(0.5),
        Dense(64, 32),
        BatchNorm0d(32),

        ReLU(),
        Dropout(0.5),
        Dense(32, 16),
        Softmax(),
    )

    indices = tf.range(batch_size, dtype=tf.int64)
    acc_to_pct = 100 / (batches_per_epoch * batch_size)
    params = list(model.each_parameter())

    for epoch_id in range(num_epochs):
        acc = 0
        for batch_id in range(batches_per_epoch):
            with tf.GradientTape() as tape:
                y_pred = model(x, True)
                z = tf.gather(tf.reshape(y_pred, [-1]), indices * 16 + y_true)
                loss = -tf.reduce_sum(tf.math.log(z))
            grads = tape.gradient(loss, params)
            for p, g in zip(params, grads):
                p.assign_sub(lr * g)
            z = tf.math.argmax(y_pred, 1)
            z = tf.equal(z, y_true)
            acc += int(tf.reduce_sum(tf.cast(z, tf.int32)))
        pct = acc * acc_to_pct
        print('%6d %6.2f' % (epoch_id, pct))


if __name__ == '__main__':
    main()
