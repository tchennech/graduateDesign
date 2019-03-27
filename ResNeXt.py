import tensorflow as tf
from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope
import numpy as np
import cv2
import os
from cifar10 import * 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
weight_decay = 0.0005
momentum = 0.9

init_learning_rate = 0.1
cardinality = 8  # how many split ?
blocks = 3  # res_block ! (split + transition)

"""
So, the total number of layers is (3*blokcs)*residual_layer_num + 2
because, blocks = split(conv 2) + transition(conv 1) = 3 layer
and, first conv layer 1, last dense layer 1
thus, total number of layers = (3*blocks)*residual_layer_num + 2
"""

depth = 64  # out channel

batch_size = 30
iteration = 238
enoughSize = 7140
# 30 * 272 ~ 8160

test_iteration = 10

total_epochs = 30


def conv_layer(input, filter, kernel, stride, padding='SAME', layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, use_bias=False, filters=filter,
                                   kernel_size=kernel, strides=stride, padding=padding)
        return network


def Global_Average_Pooling(x):
    return global_avg_pool(x, name='Global_avg_pooling')


def Average_pooling(x, pool_size=[2, 2], stride=2, padding='SAME'):
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


def Batch_Normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True):
        return tf.cond(training,
                       lambda: batch_norm(
                           inputs=x, is_training=training, reuse=None),
                       lambda: batch_norm(inputs=x, is_training=training, reuse=True))


def Relu(x):
    return tf.nn.relu(x)


def Concatenation(layers):
    return tf.concat(layers, axis=3)


def Linear(x):
    return tf.layers.dense(inputs=x, use_bias=False, units=class_num, name='linear')


def Evaluate(sess):
    test_acc = 0.0
    test_loss = 0.0
    test_pre_index = 0
    add = 200

    for it in range(test_iteration):
        test_batch_x = test_x[test_pre_index: test_pre_index + add]
        test_batch_y = test_y[test_pre_index: test_pre_index + add]
        test_pre_index = test_pre_index + add

        test_feed_dict = {
            x: test_batch_x,
            label: test_batch_y,
            learning_rate: epoch_learning_rate,
            training_flag: False
        }

        loss_, acc_ = sess.run([cost, accuracy], feed_dict=test_feed_dict)

        test_loss += loss_
        test_acc += acc_

    test_loss /= test_iteration  # average loss
    test_acc /= test_iteration  # average accuracy

    summary = tf.Summary(value=[tf.Summary.Value(tag='test_loss', simple_value=test_loss),
                                tf.Summary.Value(tag='test_accuracy', simple_value=test_acc)])

    return test_acc, test_loss, summary


class ResNeXt():
    def __init__(self, x, training):
        self.training = training
        self.model = self.Build_ResNext(x)

    def first_layer(self, x, scope):
        with tf.name_scope(scope):
            x = conv_layer(x, filter=64, kernel=[
                           3, 3], stride=1, layer_name=scope+'_conv1')
            x = Batch_Normalization(
                x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)

            return x

    def transform_layer(self, x, stride, scope):
        with tf.name_scope(scope):
            x = conv_layer(x, filter=depth, kernel=[
                           1, 1], stride=stride, layer_name=scope+'_conv1')
            x = Batch_Normalization(
                x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)

            x = conv_layer(x, filter=depth, kernel=[
                           3, 3], stride=1, layer_name=scope+'_conv2')
            x = Batch_Normalization(
                x, training=self.training, scope=scope+'_batch2')
            x = Relu(x)
            return x

    def transition_layer(self, x, out_dim, scope):
        with tf.name_scope(scope):
            x = conv_layer(x, filter=out_dim, kernel=[
                           1, 1], stride=1, layer_name=scope+'_conv1')
            x = Batch_Normalization(
                x, training=self.training, scope=scope+'_batch1')
            # x = Relu(x)

            return x

    def split_layer(self, input_x, stride, layer_name):
        with tf.name_scope(layer_name):
            layers_split = list()
            for i in range(cardinality):
                splits = self.transform_layer(
                    input_x, stride=stride, scope=layer_name + '_splitN_' + str(i))
                layers_split.append(splits)

            return Concatenation(layers_split)

    def residual_layer(self, input_x, out_dim, layer_num, res_block=blocks):
        # split + transform(bottleneck) + transition + merge

        for i in range(res_block):
            # input_dim = input_x.get_shape().as_list()[-1]
            input_dim = int(np.shape(input_x)[-1])

            if input_dim * 2 == out_dim:
                flag = True
                stride = 2
                channel = input_dim // 2
            else:
                flag = False
                stride = 1
            x = self.split_layer(input_x, stride=stride,
                                 layer_name='split_layer_'+layer_num+'_'+str(i))
            x = self.transition_layer(
                x, out_dim=out_dim, scope='trans_layer_'+layer_num+'_'+str(i))

            if flag is True:
                pad_input_x = Average_pooling(input_x)
                # [?, height, width, channel]
                pad_input_x = tf.pad(
                    pad_input_x, [[0, 0], [0, 0], [0, 0], [channel, channel]])
            else:
                pad_input_x = input_x

            input_x = Relu(x + pad_input_x)

        return input_x

    def Build_ResNext(self, input_x):
        # only cifar10 architecture

        input_x = self.first_layer(input_x, scope='first_layer')

        x = self.residual_layer(input_x, out_dim=64, layer_num='1')
        x = self.residual_layer(x, out_dim=128, layer_num='2')
        x = self.residual_layer(x, out_dim=256, layer_num='3')

        x = Global_Average_Pooling(x)
        x = flatten(x)
        x = Linear(x)

        # x = tf.reshape(x, [-1,10])
        return x

def readImage(dir):
    totalImage =  []
    totalFlag = []
    for i in range(8):
        dirName = "10"+str(i+1)
        if i == 7:
            dirName = "102"
        realPath = dir + "/" + dirName
        for fileName in os.listdir(realPath):
            img = cv2.imread(realPath+"/"+fileName)
            if img is None:
                continue
            totalImage.append(img)
            totalFlag.append([i])
    totalImage =  np.array(totalImage)
    totalFlag = np.array(totalFlag)
    return totalImage, totalFlag


path = "./data/Category-"+"32"
totalImage, totalFlag = readImage(path)
length = 7160
train_x, train_y = totalImage[:length], totalFlag[:length]
test_x, test_y = totalImage[length:], totalFlag[:length:]
image_size = 32
img_channels = 3
class_num = 1 #实际8
x = tf.placeholder(tf.float32, shape=[
                None, image_size, image_size, img_channels])
label = tf.placeholder(tf.float32, shape=[None, class_num])

training_flag = tf.placeholder(tf.bool)
logits = ResNeXt(x, training=training_flag).model
learning_rate = tf.placeholder(tf.float32, name='learning_rate')
epoch_learning_rate = init_learning_rate
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
      logits=logits, labels=label))
l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
optimizer = tf.train.MomentumOptimizer(
    learning_rate=learning_rate, momentum=momentum, use_nesterov=True)
train = optimizer.minimize(cost + l2_loss * weight_decay)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  
saver = tf.train.Saver(tf.global_variables())
def main():  
  global epoch_learning_rate
  with tf.Session() as sess:
      ckpt = tf.train.get_checkpoint_state('./model')
      if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
          saver.restore(sess, ckpt.model_checkpoint_path)
      else:
          sess.run(tf.global_variables_initializer())

      summary_writer = tf.summary.FileWriter('./logs', sess.graph)

      
      for epoch in range(1, total_epochs + 1):
          if epoch == (total_epochs * 0.5) or epoch == (total_epochs * 0.75):
              epoch_learning_rate = epoch_learning_rate / 10

          pre_index = 0
          train_acc = 0.0
          train_loss = 0.0

          for step in range(1, iteration):
              if pre_index + batch_size < enoughSize:
                  batch_x = train_x[pre_index: pre_index + batch_size]
                  batch_y = train_y[pre_index: pre_index + batch_size]
              else:
                  batch_x = train_x[pre_index:]
                  batch_y = train_y[pre_index:]
              print(pre_index, batch_size)
              batch_x = data_augmentation(batch_x)
              train_feed_dict = {
                  x: batch_x,
                  label: batch_y,
                  learning_rate: epoch_learning_rate,
                  training_flag: True
              }

              _, batch_loss = sess.run([train, cost], feed_dict=train_feed_dict)
              batch_acc = accuracy.eval(feed_dict=train_feed_dict)

              train_loss += batch_loss
              train_acc += batch_acc
              pre_index += batch_size

          train_loss /= iteration  # average loss
          train_acc /= iteration  # average accuracy

          train_summary = tf.Summary(value=[tf.Summary.Value(tag='train_loss', simple_value=train_loss),
                                            tf.Summary.Value(tag='train_accuracy', simple_value=train_acc)])

          test_acc, test_loss, test_summary = Evaluate(sess)

          summary_writer.add_summary(summary=train_summary, global_step=epoch)
          summary_writer.add_summary(summary=test_summary, global_step=epoch)
          summary_writer.flush()

          line = "epoch: %d/%d, train_loss: %.4f, train_acc: %.4f, test_loss: %.4f, test_acc: %.4f \n" % (
              epoch, total_epochs, train_loss, train_acc, test_loss, test_acc)
          print(line)

          with open('logs.txt', 'a') as f:
              f.write(line)

          saver.save(sess=sess, save_path='./model/ResNeXt.ckpt')
if __name__ == '__main__':
    print(totalFlag.shape)
    main()
    