import tensorflow as tf
import math
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def conv_layer(input,channels_in,channels_out,strides, name="conv",):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([5,5,channels_in,channels_out],stddev=0.1),name="W")
        b = tf.Variable(tf.ones([channels_out])/10,name="B")
        conv = tf.nn.conv2d(input=input,filter=w,strides=[1,strides,strides,1],padding="SAME")
        # print conv
        act = tf.nn.relu(conv + b)
        tf.summary.histogram("weight",w)
        tf.summary.histogram("bias",b)
        tf.summary.histogram("activations",act)
        # print act
        return act

def fc_layer(input,channels_in,channels_out, name="fc"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([channels_in,channels_out],stddev=0.1),name="W")
        b = tf.Variable(tf.ones([channels_out])/10,name="B")
        print input
        act = tf.nn.relu(tf.matmul(input, w) + b)
        # act = tf.matmul(input, w) + b
        return act

def compatible_convolutional_noise_shape(Y):
    noiseshape = tf.shape(Y)
    noiseshape = noiseshape * tf.constant([1,0,0,1]) + tf.constant([0,1,1,0])
    return noiseshape

x = tf.placeholder(dtype=tf.float32,shape=[None,784],name="x")
y = tf.placeholder(dtype=tf.float32,shape=[None,10],name="lables")

# learning rate
lr = tf.placeholder(tf.float32)
# dropout
pkeep = tf.placeholder(tf.float32)
pkeep_conv = tf.placeholder(tf.float32)
# batch_normal
tst = tf.placeholder(tf.bool)
iter = tf.placeholder(tf.float32)


x_image = tf.reshape(x,shape=[-1,28,28,1])
tf.summary.image("input",x_image,6)

# creat net
conv1 = conv_layer(x_image,1,4,1,"conv_1")
# conv1_dr = tf.nn.dropout(conv1,pkeep_conv,compatible_convolutional_noise_shape(conv1))
# pool1 = tf.nn.max_pool(value=conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

conv2 = conv_layer(conv1,4,8,2,"conv_2")
# conv1_dr = tf.nn.dropout(conv1,pkeep_conv,compatible_convolutional_noise_shape(conv1))
# pool2 = tf.nn.max_pool(value=conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
conv3 = conv_layer(conv2,8,16,2,"conv_3")

flattened = tf.reshape(conv3,[-1, 7 * 7 * 16])
fc1 = fc_layer(flattened,7 * 7 * 16, 200,"fc1")
# fc1_dr = tf.nn.dropout(fc1,pkeep)

w5 = tf.Variable(tf.truncated_normal([200,10],stddev=0.1))
b5 = tf.Variable(tf.ones([10])/10)
logits = tf.matmul(fc1,w5)+b5

# three convolutional layers with their channel counts, and a
# fully connected layer (tha last layer has 10 softmax neurons)
# K = 4  # first convolutional layer output depth
# L = 8  # second convolutional layer output depth
# M = 12  # third convolutional layer
# N = 200  # fully connected layer
#
# W1 = tf.Variable(tf.truncated_normal([5, 5, 1, K], stddev=0.1))  # 5x5 patch, 1 input channel, K output channels
# B1 = tf.Variable(tf.ones([K])/10)
# W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1))
# B2 = tf.Variable(tf.ones([L])/10)
# W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1))
# B3 = tf.Variable(tf.ones([M])/10)
#
# W4 = tf.Variable(tf.truncated_normal([7 * 7 * M, N], stddev=0.1))
# B4 = tf.Variable(tf.ones([N])/10)
# W5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))
# B5 = tf.Variable(tf.ones([10])/10)
#
# # The model
# stride = 1  # output is 28x28
# Y1 = tf.nn.relu(tf.nn.conv2d(x_image, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
# stride = 2  # output is 14x14
# Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2)
# stride = 2  # output is 7x7
# Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)
#
# # reshape the output from the third convolution for the fully connected layer
# YY = tf.reshape(Y3, shape=[-1, 7 * 7 * M])
#
# Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4)
# logits = tf.matmul(Y4, W5) + B5


# logits = fc_layer(fc1_dr, 1000, 10,"fc2")
# logits = tf.nn.softmax(logits)

with tf.name_scope("loss"):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=logits))
    tf.summary.scalar("loss",cross_entropy)
with tf.name_scope("BP"):
    train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(logits,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,dtype=tf.float32))
    tf.summary.scalar("accuracy",accuracy)

# writer = tf.summary.FileWriter("~/workspace/TensorFlow_Test/1")

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter("/tmp/mnist_demo/4")
    writer.add_graph(sess.graph)

    # learning rate decay
    max_learning_rate = 0.02
    min_learning_rate = 0.001
    decay_speed = 2000
    for i in range(2001):
        batch = mnist.train.next_batch(100)

        learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-1/decay_speed)
        if i % 5 == 0:
            s = sess.run(merged_summary,feed_dict={x:batch[0],y:batch[1],lr:learning_rate,pkeep:0.9,pkeep_conv:0.9})
            writer.add_summary(s,i)

        if i  % 100 == 0:
            [train_accuracy] = sess.run([accuracy], feed_dict={x: mnist.test.images, y: mnist.test.labels,lr:learning_rate,pkeep:1.0,pkeep_conv:1.0})
            print("step %d, test accuracy is %g" %(i,train_accuracy))
        sess.run(train_step,feed_dict={x:batch[0],y:batch[1],lr:learning_rate,pkeep:0.9,pkeep_conv:0.9})
