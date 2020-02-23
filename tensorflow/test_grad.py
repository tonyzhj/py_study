import tensorflow as tf

x = tf.constant(3.0)
with tf.GradientTape() as g:
  g.watch(x)
  y = x ** 3
dy_dx = g.gradient(y, x) # y’ = 2*x = 2*3 = 6

print(dy_dx)

print(list(zip([[2, 3], [4, 5], [8, 9]], [3, 6, 9])))

print([1, 2])
print(type(([1, 2], [3, 4])))
print(type([[1, 2], [3, 4]]))
