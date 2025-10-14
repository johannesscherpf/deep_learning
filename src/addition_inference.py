import numpy as np
import keras

# inference
# load model
loaded_model = keras.models.load_model('model_addition.keras')

# constant test set to finally evaluate a model
test_x = np.array([(1, 0), (1, 1), (1, 2), (1, 3), (1, 4)])
test_y = np.array((1, 2, 3, 4, 5))
prediction = loaded_model.predict(test_x)
print('inference', prediction)

# calculation of test_loss
test_result = loaded_model.evaluate(test_x, test_y)
print('inference result', test_result)