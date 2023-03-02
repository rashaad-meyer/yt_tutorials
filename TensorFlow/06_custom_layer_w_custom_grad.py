import tensorflow as tf


class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, input_size, hidden_size):
        super(MyDenseLayer, self).__init__()
        self.w = None
        self.b = None
        self.hidden_size = hidden_size

        self.w = self.add_weight(
            name='w',
            shape=(input_size, self.hidden_size),
            initializer='random_normal',
            trainable=True
        )
        self.b = self.add_weight(
            name='b',
            shape=(self.hidden_size,),
            initializer='random_normal',
            trainable=True
        )

    def call(self, x):
        return custom_operation(self.w, self.b, x)


@tf.custom_gradient
def custom_operation(w, b, x):
    y = tf.matmul(x, w) + b

    def grad_fn(dldz):
        wt = tf.transpose(w)
        xt = tf.transpose(x)

        dldw = tf.matmul(xt, dldz)
        dldx = tf.matmul(dldz, wt)

        dldb = tf.reduce_sum(dldz, axis=0)

        return dldw, dldb, dldx

    return y, grad_fn


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28 * 28).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28 * 28).astype("float32") / 255.0

    print('Custom Dense Layer')
    print('-' * 30)

    model_0 = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28 * 28)),
        MyDenseLayer(28 * 28, 64),
        MyDenseLayer(64, 32),
        MyDenseLayer(32, 10)
    ])

    model_0.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"],
    )

    model_0.fit(x_train, y_train, epochs=5)
    model_0.evaluate(x_test, y_test)
    print('=' * 30)
    print('TensorFlow Dense Layer')
    print('-' * 30)

    model_1 = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28 * 28)),
        tf.keras.layers.Dense(28 * 28),
        tf.keras.layers.Dense(64),
        tf.keras.layers.Dense(32)
    ])

    model_1.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"],
    )

    model_1.fit(x_train, y_train, epochs=5)
    model_1.evaluate(x_test, y_test)
    print('=' * 30)
