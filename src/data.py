import os
import cv2
import pandas as pd

def read_dataset_markup(path):
    markup = list()
    with open(path, 'r') as file_stream:
        for line in file_stream:
            line = line.replace(' ', '').replace(';;', ';').replace('\t', '').replace('\n', '').split(';')[-3:]
            if len(line) < 3:
                continue

            line[2] = line[2].split('/')[0]
            markup.append({'Start': line[0], 'End': line[1], 'Label': line[2]})

    markup = pd.DataFrame(markup, columns=['Start', 'End', 'Label'])
    return markup

def split_to_clips(frame_sequence, clip_size, step_size):
    start_indexes = list(range(0, len(frame_sequence), step_size))
    clips = list()
    [clips.append(frame_sequence[start_index : start_index+clip_size]) for start_index in start_indexes]
    clips = [clip for clip in clips if len(clip) == clip_size]

    return clips

def train_valid_test(clips, train_fraction, valid_fraction):
    train_len = int(len(clips) * train_fraction)
    valid_len = int(len(clips) * valid_fraction)

    train_clips = clips[:train_len]
    valid_clips = clips[train_len: train_len + valid_len]
    test_clips = clips[train_len + valid_len:]

    return (train_clips, valid_clips, test_clips)

def video2frames(path_to_video, folder_path):
    vidcap = cv2.VideoCapture(path_to_video)
    success, image = vidcap.read()

    count = 0
    while success:
        image_name = str(count).zfill(8)
        cv2.imwrite(os.path.join(folder_path, '{0}.jpg'.format(image_name)), image)
        success, image = vidcap.read()
        print('Read a {0} frame: {1}'.format(count, success))
        count += 1