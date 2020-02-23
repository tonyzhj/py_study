import tensorflow as tf

a = tf.constant([1, 2, 4])
b = tf.constant([4, 5, 6])
c = a + b
print('hello')
print(c)

version = tf.__version__
gpu_ok = tf.test.is_gpu_available()
print("tf version:",version,"\nif use GPU",gpu_ok)