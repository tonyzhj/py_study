from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(32, input_shape=(500,)))
model.add(Dense(32))
print(model.weights)  # returns list of length 4