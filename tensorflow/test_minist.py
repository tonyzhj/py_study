import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0

''' 
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
    ])
 
model.compile(optimizer= tf.keras.optimizers.SGD(learning_rate=0.1),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10)
#model.evaluate(x_test, y_test)
model.save('C:\\Users\\Tony\\Downloads\\python\\tensorflow\\my_model_test.h5')
'''

model = tf.keras.models.load_model('C:\\Users\\Tony\\Downloads\\python\\py_study\\tensorflow\\my_model_test.h5')

for i in range(1, 10):
    result = model.predict(np.reshape(x_test[i], (1, 28, 28)), batch_size=1)
    predict = np.argmax(result,axis=1)  #axis = 1是取行的最大值的索引，0是列的最大值的索引
    plt.subplot(3, 3, i)
    plt.title(predict[0])
    plt.imshow(x_test[i])

plt.show()
