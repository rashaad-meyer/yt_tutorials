import tensorflow as tf
from tensorflow import keras
from keras import layers
import tensorflow_hub as hub

base_dir = '../../../data/natural_scenes/'
train_dir = base_dir + 'seg_train'
test_dir = base_dir + 'seg_test'

MODEL_URL = 'https://tfhub.dev/google/imagenet/resnet_v2_50/classification/5'

train_ds = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                       validation_split=0.2,
                                                       subset='training',
                                                       image_size=(224, 224),
                                                       seed=42,
                                                       batch_size=32)

val_ds = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                     validation_split=0.2,
                                                     subset='validation',
                                                     image_size=(224, 224),
                                                     seed=42,
                                                     batch_size=32)

test_ds = tf.keras.utils.image_dataset_from_directory(test_dir,
                                                      image_size=(224, 224),
                                                      batch_size=32)

# trainable = True for fine tuning
# trainable = False for feature extraction
model = keras.Sequential([
    layers.Input(shape=(224, 224, 3)),
    layers.Rescaling(1/255.0),
    hub.KerasLayer(MODEL_URL, trainable=False),
    layers.Dense(6, activation='softmax')
])

model.compile(loss=keras.losses.SparseCategoricalCrossentropy(),
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

model.fit(train_ds, validation_data=val_ds, epochs=3)
