import os
from wurlitzer import pipes
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
with pipes() as (out, err):
    import tensorflow as tf

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

    indices = tf.range(batch_size, dtype=tf.int64)
    samples_per_epoch = batches_per_epoch * batch_size
    params = list(model.each_parameter())

    for epoch_id in range(num_epochs):
        acc = 0
        for batch_id in range(batches_per_epoch):
            with tf.GradientTape() as tape:
                y_pred = model(x, True)
                z = tf.gather(tf.reshape(y_pred, [-1]),
                              indices * out_dim + y_true)
                loss = -tf.reduce_sum(tf.math.log(z))
            grads = tape.gradient(loss, params)
            for p, g in zip(params, grads):
                p.assign_sub(lr * g)
            z = tf.math.argmax(y_pred, 1)
            z = tf.equal(z, y_true)
            acc += int(tf.reduce_sum(tf.cast(z, tf.int32)))
        acc = 100.0 * acc / samples_per_epoch
        print('%6d %6.2f' % (batch_id, acc))


if __name__ == '__main__':
    main()
