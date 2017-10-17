import tensorflow as tf
import tensorflow.contrib.layers as l
import numpy as np
from sklearn.preprocessing import normalize

def optimize(pred, target):

    loss = tf.reduce_sum(tf.pow(pred - target, 2))
    optimizer = tf.train.AdamOptimizer(learning_rate=9e-5)
    loss_op = optimizer.minimize(loss)

    return loss_op, loss

def nn(input, targets, isTrain=True):

    with tf.variable_scope("layers", reuse=None):

        layer = l.fully_connected(input, 30)
        # for k in range(15):
        #     layer = l.fully_connected(layer, 30)#, activation_fn=None)
        #     # layer = tf.nn.dropout(layer, 0.3)

        layer = l.fully_connected(layer, 30, normalizer_fn=tf.layers.batch_normalization)
        # layer = tf.nn.dropout(layer, 0.3)
        layer = l.fully_connected(layer, 30, normalizer_fn=tf.layers.batch_normalization)
        layer = l.fully_connected(layer, 30, normalizer_fn=tf.layers.batch_normalization)#, activation_fn=None)
        layer = l.fully_connected(layer, 30, activation_fn=None)
        layer = l.fully_connected(layer, 30, normalizer_fn=tf.layers.batch_normalization)
        layer = l.fully_connected(layer, 30, normalizer_fn=tf.layers.batch_normalization)
        layer = tf.nn.dropout(layer, 0.5)
        layer = l.fully_connected(layer, 30, normalizer_fn=tf.layers.batch_normalization)
        # layer = tf.nn.dropout(layer, 0.5)
        layer = l.fully_connected(layer, 30, normalizer_fn=tf.layers.batch_normalization)
        layer = l.fully_connected(layer, 30, normalizer_fn=tf.layers.batch_normalization)#, activation_fn=None)
        layer = l.fully_connected(layer, 30, normalizer_fn=tf.layers.batch_normalization)
        layer = l.fully_connected(layer, 30, normalizer_fn=tf.layers.batch_normalization)
        layer = l.fully_connected(layer, 30, normalizer_fn=tf.layers.batch_normalization)
        layer = l.fully_connected(layer, 30, activation_fn=None)
        layer = l.fully_connected(layer, 2, activation_fn=None)
        # layer = tf.nn.softmax(layer)
        prediction = layer

        return prediction


def accuracy_calc(output_scores, ground_truth):

    count=0
    for i in range(len(output_scores)):
        if output_scores[i][0] == min(output_scores[i]):
            output_scores[i][0] = 0
            output_scores[i][1] = 1
        else:
            output_scores[i][0] = 1
            output_scores[i][1] = 0

        if output_scores[i][0] == ground_truth[i][0]:
            count+=1


    return count/len(output_scores)



stats = input()
stats = stats.split(" ")
num_train, num_features = int(stats[0]), int(stats[1])
count = 0

features = {}
labels = {}
test_features = {}
test_labels = {}


for i in range(num_train):
    line = input()
    ls = line.split(" ")

    features[ls[0]] = []
    if ls[1] == "+1":
        labels[ls[0]] = [0.2, 0.8]
    else:
        labels[ls[0]] = [0.8, 0.2]

    for i in range(2, num_features+2):
        feature = ls[i]
        features[ls[0]].append(float(feature.split(":")[1]))

    count += 1

queries = int(input())

count =0

for i in range(queries):
    line = input()
    ls = line.split(" ")

    test_features[ls[0]] = []


    for i in range(1, num_features+1):
        feature = ls[i]
        test_features[ls[0]].append(float(feature.split(":")[1]))


keys = list(features.keys())
test_keys = list(test_features.keys())

for iter in range(num_features):
    ls = []
    iter_feature = [[z[iter] for z in list(features.values())]]
    normalized_feature = normalize(iter_feature)
    ls.extend(normalized_feature[0])

    for id in range(num_train):
        features[keys[id]][iter] = ls[id]

for iter in range(num_features):
    ls = []
    iter_feature = [[z[iter] for z in list(test_features.values())]]
    normalized_feature = normalize(iter_feature)
    ls.extend(normalized_feature[0])

    for id in range(queries):
        test_features[test_keys[id]][iter] = ls[id]



with tf.Graph().as_default() :

    sess = tf.Session()

    epochs = 800


    x = tf.placeholder(tf.float32,[None,num_features])
    y = tf.placeholder(tf.float32,[None,2])
    reuse = tf.placeholder(tf.bool)
    prediction = nn(x,y)
    loss_step, loss = optimize(prediction,y)

    sess.run(tf.global_variables_initializer())

    for batch in range(int(num_train/100)):

        x_batch = np.array([features[x] for x in keys[batch * 100:batch * 100 + 100]])
        y_batch = np.array([labels[x] for x in keys[batch * 100:batch * 100 + 100]])

        tot_loss = 0.0
        for epoch in range(epochs):
            ls, _= sess.run([loss, loss_step], feed_dict={x:x_batch, y:y_batch})

    for k in test_features:
        pred = sess.run(prediction, feed_dict={x:[test_features[k]],y:[[None, None]]})
        if pred[0][0] > pred[0][1]:
            label = "-1"
        else:
            label = "+1"

        print(k, " ", label)

