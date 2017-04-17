import tensorflow as tf
import numpy as np

#目前tensorflow只能保存变量，不能将整个神经网络保存下来

'''
##Save Weight and Biases to file##
W=tf.Variable([[1,2,3],[3,4,5]],dtype=tf.float32)
b=tf.Variable([[1,2,3]],dtype=tf.float32)

init = tf.global_variables_initializer()

saver=tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    save_path=saver.save(sess,"test/save_net.ckpt")
    print("Save to path:",save_path)
'''

'''
##Get Weight and Biases from file##
W=tf.Variable(np.arange(6).reshape((2,3)),dtype=tf.float32)
b=tf.Variable(np.arange(3).reshape((1,3)),dtype=tf.float32)

saver=tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess,"test/save_net.ckpt")
    print(sess.run(W))
    print(sess.run(b))
'''
