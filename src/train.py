import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import random
import tensorflow as tf

from data import read_dataset_markup, split_to_clips, train_valid_test
from generator import DataGenerator
from model import get_c3d_graph_nodes, save_model

DATA_FOLDER = r'../data/'
MODELS_FOLDER = r'../models/'
BATCH_SIZE = 10
CLIP_SIZE = 25
HEIGHT = 112
WIDTH = 112
EPOCHS = 2


def loop(sess, data_generator, train, mask, graph_nodes):
    total_loss = 0.
    total_accuracy = 0.

    for batch_index in range(len(data_generator)):
        x_batch, y_batch = data_generator[batch_index]

        if train:
            _, batch_loss, batch_accuracy = sess.run([graph_nodes['optimizer'],
                                                      graph_nodes['loss'],
                                                      graph_nodes['accuracy']],
                                                     feed_dict={graph_nodes['X']: x_batch,
                                                                graph_nodes['y']: y_batch,
                                                                graph_nodes['is_training']: train})
        else:
            batch_loss, batch_accuracy = sess.run([graph_nodes['loss'], graph_nodes['accuracy']],
                                                  feed_dict={graph_nodes['X']: x_batch,
                                                             graph_nodes['y']: y_batch,
                                                             graph_nodes['is_training']: train})

        total_loss += batch_loss
        total_accuracy += batch_accuracy

        if train:
            print('{0}/{1} - loss: {2} - acc: {3}'.format(
                batch_index + 1,
                len(data_generator),
                round(total_loss / (batch_index + 1), 4),
                round(total_accuracy / (batch_index + 1), 4)), end='\r')

    if not train:
        print('{2}_loss: {0} - {2}_acc: {1}'.format(
            round(total_loss / len(data_generator), 4),
            round(total_accuracy / len(data_generator), 4),
            mask)
        )


if __name__ == '__main__':
    markup = read_dataset_markup(os.path.join(DATA_FOLDER, 'markup.txt'))

    # Preparation
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

    # Network
    graph_nodes = get_c3d_graph_nodes(CLIP_SIZE, HEIGHT, WIDTH)
    saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2)

    #
    clips = split_to_clips(frames, clip_size=CLIP_SIZE, step_size=25)
    random.shuffle(clips)
    train_clips, valid_clips, test_clips = train_valid_test(clips, 0.7, 0.2)

    base_params = [labels, BATCH_SIZE, CLIP_SIZE, HEIGHT, WIDTH, os.path.join(DATA_FOLDER, 'frames')]
    train_data_generator = DataGenerator(train_clips, *base_params)
    valid_data_generator = DataGenerator(valid_clips, *base_params)
    test_data_generator = DataGenerator(test_clips, *base_params)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(EPOCHS):
            print('\nEpoch {0}/{1}'.format(epoch + 1, EPOCHS))

            train_data_generator.shuffle()
            loop(sess, train_data_generator, True, None, graph_nodes)
            loop(sess, valid_data_generator, False, 'val', graph_nodes)
            save_model(
                sess,
                path=os.path.join(MODELS_FOLDER, 'c3d_model_{0}/'.format(epoch + 1)),
                inputs={'X': graph_nodes['X'], 'is_training': graph_nodes['is_training']},
                outputs={'probabilities': graph_nodes['probabilities']}
            )

        loop(sess, test_data_generator, False, 'test', graph_nodes)