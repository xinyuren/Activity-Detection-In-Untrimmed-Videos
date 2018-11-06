import os
import random
import numpy as np
import jpeg4py as jpeg
from random import shuffle
from scipy.misc import imresize

random.seed(17)

class DataGenerator(object):

    def __init__(self, clips, labels, batch_size, clip_size, height, width, frames_folder):
        self.clips = clips
        self.labels = labels
        self.batch_size = batch_size
        self.clip_size = clip_size
        self.height = height
        self.width = width
        self.frames_folder = frames_folder

    def __len__(self):
        return int(np.ceil(len(self.clips) / float(self.batch_size))) - 1

    def __getitem__(self, index):
        batch_clips = self.clips[index*self.batch_size : (index+1)*self.batch_size]
        x_batch = np.empty((self.batch_size, self.clip_size, self.height, self.width, 3), dtype=np.uint8)
        y_batch = np.empty((self.batch_size, 1))

        for i in range(len(batch_clips)):
            clip_labels = list()

            for frame_index, frame_name in enumerate(batch_clips[i]):
                frame = jpeg.JPEG(os.path.join(self.frames_folder, frame_name)).decode()
                frame = imresize(frame, (self.height, self.width))
                x_batch[i, frame_index, ...] = frame
                clip_labels.append(self.labels[frame_name])

            y_batch[i, 0] = max(set(clip_labels), key=clip_labels.count)

        return x_batch / 255., y_batch

    def shuffle(self):
        shuffle(self.clips)