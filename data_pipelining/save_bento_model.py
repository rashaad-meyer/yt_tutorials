import keras
import bentoml

MODEL_PATH = '../models/mnist_digits_model'

model = keras.models.load_model(MODEL_PATH)
bento_model = bentoml.keras.save_model("mnist_digits_model", model)

print(f'Bento model tag = {bento_model.tag}')


