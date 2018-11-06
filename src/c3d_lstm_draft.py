import os
import random
import pandas as pd
import numpy as np
import tensorflow as tf
from scipy.misc import imresize
import jpeg4py as jpeg

DATA_FOLDER = r'../data/'
BATCH_SIZE = 4
NUMBERS_OF_CLIPS = 8
CLIP_SIZE = 16

HEIGHT = 112
WIDTH = 112
# random.seed(17)

markup = list()
with open(os.path.join(DATA_FOLDER, 'markup.txt'), 'r') as file_stream:
    for line in file_stream:
        line = line.replace(' ', '').replace(';;', ';').replace('\t', '').replace('\n', '').split(';')[-3:]
        if len(line) < 3:
            continue

        line[2] = line[2].split('/')[0]
        markup.append({'Start': line[0], 'End': line[1], 'Label': line[2]})

markup = pd.DataFrame(markup, columns=['Start', 'End', 'Label'])

action_frames = set()
for index, row in markup.iterrows():
    part_of_action_frames = range(int(row['Start']) - 1, int(row['End']) - 1)
    action_frames.update(part_of_action_frames)

labels = dict()
frames = os.listdir(os.path.join(DATA_FOLDER, 'frames'))
for frame in frames:
    if int(frame.split('.')[0]) in action_frames:
        labels[frame] = 1
    else:
        labels[frame] = 0

train_frames = frames[10000:]
test_frames = frames[:10000]


def get_batch():
    x_batch = np.empty((BATCH_SIZE, NUMBERS_OF_CLIPS, CLIP_SIZE, HEIGHT, WIDTH, 3))
    y_batch = np.empty((BATCH_SIZE, NUMBERS_OF_CLIPS, 1))
    upper_bound = len(train_frames) - NUMBERS_OF_CLIPS * CLIP_SIZE - 1

    for batch_index in range(BATCH_SIZE):
        start_frame_index = random.randint(0, upper_bound)
        frame_names = train_frames[start_frame_index: start_frame_index + NUMBERS_OF_CLIPS * CLIP_SIZE]
        clips = np.array_split(frame_names, NUMBERS_OF_CLIPS)

        for clip_index in range(NUMBERS_OF_CLIPS):
            clip_labels = list()
            for frame_index in range(CLIP_SIZE):
                frame_name = clips[clip_index][frame_index]
                frame = jpeg.JPEG(os.path.join(DATA_FOLDER, 'frames', frame_name)).decode()
                frame = imresize(frame, (HEIGHT, WIDTH))
                x_batch[batch_index, clip_index, frame_index, ...] = frame
                clip_labels.append(labels[frame_name])

            y_batch[batch_index, clip_index, 0] = max(set(clip_labels), key=clip_labels.count)

    return x_batch / 255., y_batch

# NETWORK
X = tf.placeholder(tf.float32, shape=(None, NUMBERS_OF_CLIPS, CLIP_SIZE, HEIGHT, WIDTH, 3))
y = tf.placeholder(tf.float32, shape=(None, NUMBERS_OF_CLIPS, 1))


def c3d_lstm(inputs, previous_prediction, previous_state):
    with tf.variable_scope('c3d_lstm', reuse=tf.AUTO_REUSE):
        conv1 = tf.layers.conv3d(
            inputs=inputs,
            filters=64,
            kernel_size=(3, 3, 3),
            padding='same',
            activation=tf.nn.relu,
            name='conv1'
        )
        pool1 = tf.layers.max_pooling3d(
            inputs=conv1,
            pool_size=(1, 2, 2),
            strides=(1, 2, 2),
            padding='valid',
            name='pool1'
        )

        conv2 = tf.layers.conv3d(
            inputs=pool1,
            filters=128,
            kernel_size=(3, 3, 3),
            padding='same',
            activation=tf.nn.relu,
            name='conv2'
        )
        pool2 = tf.layers.max_pooling3d(
            inputs=conv2,
            pool_size=(2, 2, 2),
            strides=(2, 2, 2),
            padding='valid',
            name='pool2'
        )

        conv3a = tf.layers.conv3d(
            inputs=pool2,
            filters=256,
            kernel_size=(3, 3, 3),
            padding='same',
            activation=tf.nn.relu,
            name='conv3a'
        )
        conv3b = tf.layers.conv3d(
            inputs=conv3a,
            filters=256,
            kernel_size=(3, 3, 3),
            padding='same',
            activation=tf.nn.relu,
            name='conv3b'
        )
        pool3 = tf.layers.max_pooling3d(
            inputs=conv3b,
            pool_size=(2, 2, 2),
            strides=(2, 2, 2),
            padding='valid',
            name='pool3'
        )

        conv4a = tf.layers.conv3d(
            inputs=pool3,
            filters=512,
            kernel_size=(3, 3, 3),
            padding='same',
            activation=tf.nn.relu,
            name='conv4a'
        )
        conv4b = tf.layers.conv3d(
            inputs=conv4a,
            filters=512,
            kernel_size=(3, 3, 3),
            padding='same',
            activation=tf.nn.relu,
            name='conv4b'
        )
        pool4 = tf.layers.max_pooling3d(
            inputs=conv4b,
            pool_size=(2, 2, 2),
            strides=(2, 2, 2),
            padding='valid',
            name='pool4'
        )

        conv5a = tf.layers.conv3d(
            inputs=pool4,
            filters=256,
            kernel_size=(3, 3, 3),
            padding='same',
            activation=tf.nn.relu,
            name='conv5a'
        )
        conv5b = tf.layers.conv3d(
            inputs=conv5a,
            filters=512,
            kernel_size=(3, 3, 3),
            padding='same',
            activation=tf.nn.relu,
            name='conv5b'
        )
        pool5 = tf.layers.max_pooling3d(
            inputs=conv5b,
            pool_size=(2, 2, 2),
            strides=(2, 2, 2),
            padding='valid',
            name='pool5'
        )

        flatten = tf.layers.flatten(
            inputs=pool5,
            name='flatten'
        )
        fc6 = tf.layers.dense(
            inputs=flatten,
            units=4096,
            activation=tf.nn.relu,
            name='fc6'
        )
        tf_is_traing_pl = tf.placeholder_with_default(True, shape=())
        do1 = tf.layers.dropout(
            inputs=fc6,
            # rate=keep_rate,
            training=tf_is_traing_pl,
            name='do1'
        )

        cell = tf.nn.rnn_cell.LSTMCell(num_units=512, state_is_tuple=True)
        if previous_state == None:
            previous_state = cell.zero_state(BATCH_SIZE, dtype=tf.float32)

        clip_features = tf.concat([previous_prediction, do1], axis=1)
        clip_features = tf.reshape(clip_features, (-1, 1, 4097))
        outputs, state = tf.nn.dynamic_rnn(cell, clip_features, initial_state=previous_state, dtype=tf.float32)
        outputs = tf.reshape(outputs, (-1, outputs.get_shape()[2]))

        p_shape = tf.shape(outputs)
        p_flat = tf.reshape(outputs, [-1])
        i_flat = tf.reshape(tf.reshape(tf.range(0, p_shape[0]) * p_shape[1], [-1, 1]) + int(outputs.get_shape()[1]) - 1, [-1])
        last_output = tf.reshape(tf.gather(p_flat, i_flat), [p_shape[0], -1])

        logit = tf.layers.dense(inputs=last_output, units=1, activation=None, name='logits')
        prediction = tf.nn.sigmoid(logit)

        return (logit, prediction, state)


previous_state = None
previous_prediction = tf.zeros((BATCH_SIZE, 1), dtype=tf.float32, name='starting_flag')
total_loss = 0.

for num_step in range(NUMBERS_OF_CLIPS):
    logit, prediction, state = c3d_lstm(X[:, num_step, :, :, :, :], previous_prediction, previous_state)

    previous_prediction = prediction
    previous_state = state

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit, labels=y[:, num_step, :]))
    total_loss += loss

total_loss = tf.divide(total_loss, NUMBERS_OF_CLIPS)
optimizer = tf.train.AdamOptimizer(0.1).minimize(total_loss)


with tf.Session() as sess:
    # loss = 0.
    sess.run(tf.global_variables_initializer())
    # x_batch, y_batch = get_batch()
    # logit, prediction, state = sess.run([logit, prediction, state], feed_dict={X: x_batch, y: y_batch})

    for i in range(1, 501):
        x_batch, y_batch = get_batch()
        batch_loss, _ = sess.run([total_loss, optimizer], feed_dict={X: x_batch, y: y_batch})
        # loss += batch_loss
        print(i, batch_loss)

    a = 4


































