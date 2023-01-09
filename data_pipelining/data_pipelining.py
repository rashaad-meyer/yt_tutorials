import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

base_path = '../../data/mnist_2_digits/'

mode = 'train'

train_df = pd.read_csv(base_path + mode + '.csv')
file_names = train_df['file_name'].apply(lambda file_name: base_path + mode + '/' + file_name)

labels = list(zip(train_df['label_1'], train_df['label_2']))

file_and_labels = (file_names, labels)


def process_path(path, label):
    img = tf.io.read_file(path)
    img = tf.io.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (64, 64))

    labels = {'first_num': label[0], 'second_num': label[1]}

    return img, labels


train_ds = tf.data.Dataset.from_tensor_slices(file_and_labels)
train_ds = train_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)

train_ds = train_ds.shuffle(buffer_size=len(train_ds))

# If you would like split the dataset up into train and test sets
# test_ds =  train_ds[len(train_ds)*0.8:]
# train_ds = train_ds[len(train_ds)*0.8:]

train_ds = train_ds.batch(64)

inputs = keras.Input(shape=(64, 64, 1))

x = layers.Conv2D(filters=32, kernel_size=3, padding='same')(inputs)
x = layers.BatchNormalization()(x)
x = keras.activations.relu(x)

x = layers.Conv2D(filters=64, kernel_size=3, padding='same')(x)
x = layers.BatchNormalization()(x)
x = keras.activations.relu(x)

x = layers.MaxPooling2D()(x)

x = layers.Conv2D(64, 3, activation='relu')(x)
x = layers.Conv2D(128, 3, activation='relu')(x)
x = layers.MaxPooling2D()(x)

x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)

x = layers.Dense(64, activation='relu')(x)

output1 = layers.Dense(10, activation='softmax', name='first_num')(x)
output2 = layers.Dense(10, activation='softmax', name='second_num')(x)

model = keras.Model(inputs=inputs, outputs=[output1, output2])

model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

model.fit(train_ds, epochs=5, verbose=2)
