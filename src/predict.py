import cv2
import os
import collections
import numpy as np
import tensorflow as tf
import jpeg4py as jpeg
from scipy.misc import imresize
from skimage.io import imsave
from tensorflow.python.saved_model import tag_constants

from data import split_to_clips, video2frames

DATA_FOLDER = r'../data/'
MODELS_FOLDER = r'../models/'
BATCH_SIZE = 10
CLIP_SIZE = 25
HEIGHT = 112
WIDTH = 112
WINDOW_SIZE = 30
THRESHOLD = 0.43

def get_batch(index, clips, batch_size, clip_size, height, width):
    batch_clips = clips[index * batch_size: (index + 1) * batch_size]
    x_batch = np.empty((batch_size, clip_size, height, width, 3), dtype=np.uint8)

    for i in range(len(batch_clips)):
        for frame_index, frame_name in enumerate(batch_clips[i]):
            frame = jpeg.JPEG(os.path.join(DATA_FOLDER, 'test_frames', frame_name)).decode()
            frame = imresize(frame, (height, width))
            x_batch[i, frame_index, ...] = frame

    return x_batch / 255.

def smoothing(x, window_size=5):
    lenght = len(x)
    s = np.arange(-window_size, lenght - window_size)
    e = np.arange(window_size, lenght + window_size)
    s[s < 0] = 0
    e[e >= lenght] = lenght - 1
    y = np.zeros(x.shape)
    for i in range(lenght):
        y[i] = np.mean(x[s[i]:e[i]], axis=0)

    return y

if __name__ == '__main__':
    video2frames(
        path_to_video=os.path.join(DATA_FOLDER, 'fight_margaret_2_24_01_2007.wmv'),
        folder_path=os.path.join(DATA_FOLDER, 'test_frames')
    )

    test_frames = os.listdir(os.path.join(DATA_FOLDER, 'test_frames'))
    test_clips = split_to_clips(test_frames, clip_size=CLIP_SIZE, step_size=1)
    number_of_test_batches = int(np.ceil(len(test_clips) / float(BATCH_SIZE))) - 1

    clips_frames_number = len(set([item for sublist in test_clips for item in sublist]))
    frame_probabilities = np.zeros((1, clips_frames_number))
    frame_frequencies = np.zeros((1, clips_frames_number))

    # Prediction
    with tf.Session() as sess:
        tf.saved_model.loader.load(sess, [tag_constants.SERVING], os.path.join(MODELS_FOLDER, 'c3d_model_2/'))
        graph = tf.get_default_graph()

        X = graph.get_tensor_by_name('X:0')
        training = graph.get_tensor_by_name('training:0')
        probabilities = graph.get_tensor_by_name('probabilities:0')

        start_index = 0
        for batch_index in range(number_of_test_batches):
            x_batch = get_batch(batch_index, test_clips, BATCH_SIZE, CLIP_SIZE, HEIGHT, WIDTH)
            batch_probas = sess.run(probabilities, feed_dict={X: x_batch, training: True})

            for batch_proba in enumerate(batch_probas):
                clip_probas = np.array([batch_proba[1][0]] * CLIP_SIZE)
                clip_freqs = np.array([1] * CLIP_SIZE)
                frame_probabilities[0, start_index : start_index + CLIP_SIZE] += clip_probas
                frame_frequencies[0, start_index: start_index + CLIP_SIZE] += clip_freqs
                start_index += 1

            print(batch_index + 1, number_of_test_batches)

        frame_probas = dict(zip(test_frames[:-5], list(frame_probabilities[0, :-5] / frame_frequencies[0, :-5])))


    # Resave
    ordered_frame_probas = collections.OrderedDict(sorted(frame_probas.items()))
    probas = list(ordered_frame_probas.values())[0::CLIP_SIZE]
    smoothed_values = smoothing(np.array(probas), WINDOW_SIZE)
    clip_labels = (smoothed_values >= THRESHOLD).astype(int)

    # x = np.array(range(smoothed_values.shape[0]))
    # p = np.poly1d(np.polyfit(x, smoothed_values, deg=100))
    # clip_labels = (p(x) >= 0.445).astype(int)

    frame_labels = list()
    for clip_label in clip_labels:
        frame_labels.extend([clip_label] * CLIP_SIZE)

    for i, frame_name in enumerate(list(ordered_frame_probas.keys())):
        print(frame_name)
        frame = jpeg.JPEG(os.path.join(DATA_FOLDER, 'test_frames', frame_name)).decode()

        if frame_labels[i] > 0:
            cv2.putText(frame, 'ACTION', (450, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, 'NO ACTION', (450, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        imsave(os.path.join(DATA_FOLDER, 'test_frame_predictions', frame_name), frame)