import bentoml

from bentoml.io import JSON, Image

import numpy as np

model_ref = bentoml.keras.get('mnist_digits_model:ww4ilpfjd2vodptb')

model_runner = model_ref.to_runner()

svc = bentoml.Service('2_digit_mnist_classifier', runners=[model_runner])


@svc.api(input=Image(), output=JSON())
def classify(application_data):
    img = np.asarray(application_data)
    img = np.expand_dims(img, axis=0)
    pred = model_runner.predict.run(img)
    pred = {'first_num': np.argmax(pred[0]),
            'second_num': np.argmax(pred[1])}
    return pred
