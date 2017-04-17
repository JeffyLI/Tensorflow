import tensorflow as tf

#Session
'''
matrix1=tf.constant([[3,3]])
matrix2=tf.constant([[2],[2]])

product=tf.matmul(matrix1,matrix2)


# method1
sess=tf.Session()
print(sess.run(product))
sess.close()


# method2

with tf.Session() as sess:
    print(sess.run(product))
'''


#Variable
'''
state=tf.Variable(0)
one=tf.Variable(1)
new_value=tf.add(state,one)
update=tf.assign(state,new_value)

init=tf.global_variables_initializer()  #must hava if define variable

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))
'''

#placeholder
input1=tf.placeholder(tf.float32)
input2=tf.placeholder(tf.float32)

output=tf.mul(input1,input2)

with tf.Session() as sess:
    print(sess.run(output,feed_dict={input1:[7.],input2:[2.]}))

