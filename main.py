import tensorflow as tf
import numpy as np
import os
from preperData import *
from ResNextNet import ResNeXt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
weight_decay = 0.0005
momentum = 0.9


cost = 0.0
accuracy = 0.0

init_learning_rate = 0.05
epoch_learning_rate = init_learning_rate
batch_size = 40
iteration = 178
enoughSize = 7140
# 40 * 178 ~ 7140

test_iteration = 10

total_epochs = 5

train_x = None
train_y = None
test_x = None
test_y = None
x = None
label = None
learning_rate = None
training_flag = None
predict = None

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


def trainModel(savePath):
    global epoch_learning_rate, cost, accuracy, train_x, train_y, test_x, test_y, x, label, learning_rate, training_flag
    # 数据集路径
    path = "./data/Category-"+"32"
    totalImage, totalFlag = readImage(path)

    # 裁剪长度，为了方便，测试集为2000
    length = 7160
    train_x, train_y = totalImage[:length], totalFlag[:length]
    test_x, test_y = totalImage[length:], totalFlag[:length:]
    image_size = 32
    img_channels = 3
    class_num = 8  # 实际8
    x = tf.placeholder(tf.float32, shape=[
        None, image_size, image_size, img_channels], name='input_x')
    label = tf.placeholder(tf.float32, shape=[None, class_num], name='input_y')

    training_flag = tf.placeholder(tf.bool, name='flag')
    logits = ResNeXt(x, training=training_flag).model
    predict = tf.argmax(logits, 1, name="predict")
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    epoch_learning_rate = init_learning_rate
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=label, logits=logits))
    l2_loss = tf.add_n([tf.nn.l2_loss(var)
                        for var in tf.trainable_variables()])

    # 正则化
    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate, momentum=momentum, use_nesterov=True)
    train = optimizer.minimize(cost + l2_loss * weight_decay)

    # 准确率
    correct_prediction = tf.equal(predict, tf.argmax(label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver(tf.global_variables())
    tf.add_to_collection("predict_collection", logits)
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
                #print(pre_index, batch_size)
                batch_x = data_augmentation(batch_x)
                train_feed_dict = {
                    x: batch_x,
                    label: batch_y,
                    learning_rate: epoch_learning_rate,
                    training_flag: True
                }

                _, batch_loss = sess.run(
                    [train, cost], feed_dict=train_feed_dict)
                batch_acc = accuracy.eval(feed_dict=train_feed_dict)

                train_loss += batch_loss
                train_acc += batch_acc
                pre_index += batch_size

            train_loss /= iteration  # average loss
            train_acc /= iteration  # average accuracy

            train_summary = tf.Summary(value=[tf.Summary.Value(tag='train_loss', simple_value=train_loss),
                                              tf.Summary.Value(tag='train_accuracy', simple_value=train_acc)])

            test_acc, test_loss, test_summary = Evaluate(sess)

            summary_writer.add_summary(
                summary=train_summary, global_step=epoch)
            summary_writer.add_summary(summary=test_summary, global_step=epoch)
            summary_writer.flush()

            line = "epoch: %d/%d, train_loss: %.4f, train_acc: %.4f, test_loss: %.4f, test_acc: %.4f \n" % (
                epoch, total_epochs, train_loss, train_acc, test_loss, test_acc)
            print(line)

            with open('logs.txt', 'a') as f:
                f.write(line)

            saver.save(sess=sess, save_path='./model/'+savePath+'.ckpt')


def readModel(name):
    datasName = input("输入数据集名字")
    X = preperDatas(datasName)
    print(X.shape)
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('./model/'+name)
        saver.restore(sess, tf.train.latest_checkpoint("./model/"))
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name('input_x:0')
        training_flag = graph.get_tensor_by_name('flag:0')
        test_feed_dict = {
            x: X,
            training_flag: False
        }
        result = graph.get_tensor_by_name("predict:0")
        pred = sess.run(result, feed_dict=test_feed_dict)
        print(pred)


if __name__ == '__main__':
    function = int(input("选择功能，0：训练；1：分类"))
    if(function):
        modelName = input("输入模型名字")
        readModel(modelName)
    else:
        modelName = input("输入模型名字")
        trainModel(modelName)
