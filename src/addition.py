import random
import keras
import numpy as np
from keras import Input
import matplotlib.pyplot as plt

# data
random.seed(42)
x_train = []
y_train = []

n = 1000

for i in range(n):
    x_0 = random.randint(0, 100)
    x_1 = random.randint(0, 100)
    y = x_0 + x_1
    x_train.append((x_0, x_1))
    y_train.append(y)

print(x_train[0])
print(y_train[0])

x_train = np.array(x_train)
y_train = np.array(y_train)

# network architecture
inputs = Input(shape=(2,))
dense_layer = keras.layers.Dense(units=1, activation='linear')(inputs)
model = keras.models.Model(inputs=inputs, outputs=dense_layer)
model.summary()
print('weights before training:', model.get_weights())

# trainer
model.compile(loss='mse', optimizer='adam')

# train and eval
history = model.fit(x_train, y_train, epochs=300, validation_split=0.2)
print('prediction 2 + 1 = ', model.predict((np.array([(2, 1)]))))

# plot
plt.plot(history.history['loss'], label = 'train loss')
plt.plot(history.history['val_loss'], label = 'validation loss')
plt.legend()
plt.show()

# save
model.save('model_addition.keras') # Modellarchitektur + Gewichte