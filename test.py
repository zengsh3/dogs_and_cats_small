import tensorflow as tf
5 hello = tf.constant('Hello, TensorFlow!')
6 sess = tf.Session()
7 print(sess.run(hello))
