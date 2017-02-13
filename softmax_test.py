import tensorflow as tf
import csv

def main():
    #get data_size
    size = 0
    with open("test_data.csv", "r") as infile:
        reader = csv.reader(infile)
        data = list(reader)
        size = len(data)

    #import data
    x = [[0 for j in range(6)] for i in range(size)]
    y_ = [[0,0] for i in range(size)]
    with open("test_data.csv", "r") as infile:
        line_num = 0
        data_reader = csv.reader(infile)
        for line in data_reader:
            #first 6 nums go in x.
            for i in range(6):
                x[line_num][i] = float(line[i])
            #last number used for y_
            if (int(line[6]) == 1):
                y_[line_num] = [0, 1]
            else:
                y_[line_num] = [1, 0]
            line_num += 1

    #W and b are the parameters being tuned
    W = tf.Variable(tf.zeros([6, 2]))
    b = tf.Variable(tf.zeros([2]))

    #y is the predicted labels
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    #session
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    sess.run(train_step)

    #evaluation
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy))

main()

