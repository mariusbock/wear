from datetime import datetime, timedelta, date
from glob import glob
import os
import re

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.io import wavfile

label_dict = {
    'jogging': 0,
    'jogging (rotating arms)': 1,
    'jogging (skipping)': 2,
    'jogging (sidesteps)': 3,
    'jogging (butt-kicks)': 4,
    'stretching (triceps)': 5,
    'stretching (lunging)': 6,
    'stretching (shoulders)': 7,
    'stretching (hamstrings)': 8,
    'stretching (lumbar rotation)': 9,
    'push-ups': 10,
    'push-ups (complex)': 11,
    'sit-ups': 12,
    'sit-ups (complex)': 13,
    'burpees': 14,
    'lunges': 15,
    'lunges (complex)': 16,
    'bench-dips': 17
}

def convert_labels_to_annotation_json(labels, sampling_rate, fps, l_dict):
    annotations = []
    curr_start_i = 0
    curr_end_i = 0
    curr_label = labels[0]

    for i, l in enumerate(labels):
        if curr_label != l:
            act_start = curr_start_i / sampling_rate
            act_end = curr_end_i / sampling_rate
            act_label = curr_label
            # create annotation
            if act_label != 'null' and not pd.isnull(act_label):
                anno = {
                    'label': act_label,
                    'segment': [
                        act_start,
                        act_end
                    ],
                    'segment (frames)': [
                        act_start * fps,
                        act_end * fps
                    ],
                    'label_id': l_dict[act_label]
                    }
                annotations.append(anno)  
            curr_label = l
            curr_start_i = i + 1
            curr_end_i = i + 1
        else:
            curr_end_i += 1
    return annotations

def apply_sliding_window(data, window_size, window_overlap):
    output_x = None
    output_y = None
    output_sbj = []
    for i, subject in enumerate(np.unique(data[:, 0])):
        subject_data = data[data[:, 0] == subject]
        subject_x, subject_y = subject_data[:, :-1], subject_data[:, -1]
        tmp_x, _ = sliding_window_samples(subject_x, window_size, window_overlap)
        tmp_y, _ = sliding_window_samples(subject_y, window_size, window_overlap)

        if output_x is None:
            output_x = tmp_x
            output_y = tmp_y
            output_sbj = np.full(len(tmp_y), subject)
        else:
            output_x = np.concatenate((output_x, tmp_x), axis=0)
            output_y = np.concatenate((output_y, tmp_y), axis=0)
            output_sbj = np.concatenate((output_sbj, np.full(len(tmp_y), subject)), axis=0)

    output_y = [[i[-1]] for i in output_y]
    return output_sbj, output_x, np.array(output_y).flatten()

def sliding_window_samples(data, win_len, overlap_ratio=None):
    """
    Return a sliding window measured in seconds over a data array.

    :param data: dataframe
        Input array, can be numpy or pandas dataframe
    :param length_in_seconds: int, default: 1
        Window length as seconds
    :param sampling_rate: int, default: 50
        Sampling rate in hertz as integer value
    :param overlap_ratio: int, default: None
        Overlap is meant as percentage and should be an integer value
    :return: tuple of windows and indices
    """
    windows = []
    indices = []
    curr = 0
    overlapping_elements = 0

    if overlap_ratio is not None:
        if not ((overlap_ratio / 100) * win_len).is_integer():
            float_prec = True
        else:
            float_prec = False
        overlapping_elements = int((overlap_ratio / 100) * win_len)
        if overlapping_elements >= win_len:
            print('Number of overlapping elements exceeds window size.')
            return
    changing_bool = True
    while curr < len(data) - win_len:
        windows.append(data[curr:curr + win_len])
        indices.append([curr, curr + win_len])
        if (float_prec == True) and (changing_bool == True):
            curr = curr + win_len - overlapping_elements - 1
            changing_bool = False
        else:
            curr = curr + win_len - overlapping_elements
            changing_bool = True

    return np.array(windows), np.array(indices)


def merge_sensor_csv(input_dir, sensor):
    filenames = sorted(glob(os.path.join(os.path.join(input_dir, 'sensor_data', sensor, '*.csv'))))
    output = pd.DataFrame()

    for i, filename in enumerate(filenames):
        curr_file = pd.read_csv(filename)
        if output is None:
            output = curr_file.iloc[:, 1:]
        else:
            output = pd.concat((output, curr_file.iloc[:, 1:]), axis=0)

    scaler = MinMaxScaler()
    output = output[['acc_x', 'acc_y', 'acc_z']]
    output['magnitude'] = np.sqrt(output['acc_x'] ** 2 + output['acc_y'] ** 2 + output['acc_y'] ** 2)
    output['magnitude'] = scaler.fit_transform(output['magnitude'].values.reshape(-1, 1))

    return output

def create_synced_files(input_dir, sensors, data_points, sync_points, final_time):
    final_dataset = None
    # please provide as right arm, right leg, left leg, left arm
    location = ['right_arm', 'right_leg', 'left_leg', 'left_arm']
    for i, sensor in enumerate(sensors):
        data = merge_sensor_csv(input_dir, sensor)
        sens_data_points = data_points[i]
        sens_sync_point = sync_points[i]
        data = data.iloc[sens_data_points[0]-1:sens_data_points[1]]
        start, end = datetime.strptime(sens_sync_point[0], '%H:%M:%S.%f'), datetime.strptime(sens_sync_point[1], '%H:%M:%S.%f')
        today = date.today()
        start = start.replace(year=today.year, month=today.month, day=today.day)
        end = end.replace(year=today.year, month=today.month, day=today.day)
        true_sampling_rate = (end - start) / data.shape[0]
        print('Original Data had sampling rate of: {}'.format(timedelta(seconds=1) / true_sampling_rate))
        times = pd.date_range(start, end + timedelta(seconds=1), periods=data.shape[0]).strftime('%H:%M:%S.%f')

        data['time'] = times
        data['time'] = pd.to_datetime(data['time']).dt.round('1ms')
        data = data.drop_duplicates(subset=['time'])
        data.set_index('time', inplace=True)
        data = data.resample('1ms').interpolate()

        new_start, new_end = datetime.strptime(final_time[0], '%H:%M:%S.%f'), datetime.strptime(final_time[1], '%H:%M:%S.%f')
        new_start = new_start.replace(year=today.year, month=today.month, day=today.day)
        new_end = new_end.replace(year=today.year, month=today.month, day=today.day)

        data = data.resample('20ms').first()
        data['time'] = data.index
        data = data.loc[new_start:new_end]
        data = data.iloc[:-1, :]

        scaler = MinMaxScaler()
        data['magnitude'] = np.sqrt(data['acc_x'] ** 2 + data['acc_y'] ** 2 + data['acc_y'] ** 2)
        data['magnitude'] = scaler.fit_transform(data['magnitude'].values.reshape(-1, 1))

        data.to_csv(os.path.join(input_dir, location[i] + '_synced.csv'), index=False)
        wavfile.write(os.path.join(input_dir, location[i] + '_acc_mag.wav'), 50, data['magnitude'])

        data = data.rename(columns={"acc_x": location[i] + '_acc_x',
                                    "acc_y": location[i] + '_acc_y',
                                    "acc_z": location[i] + '_acc_z'})
        if final_dataset is None:
            final_dataset = data[[location[i] + '_acc_x', location[i] + '_acc_y', location[i] + '_acc_z']]
        else:
            final_dataset = pd.concat((final_dataset, data[[location[i] + '_acc_x', location[i] + '_acc_y', location[i] + '_acc_z']]),
                                      axis=1)
    final_dataset.to_csv(os.path.join(input_dir, input_dir.split('/')[-1] + '.csv'), index=False)


def append_srt_to_dataframe(data, path_to_srt, sampling_rate):
    with open(path_to_srt, 'r') as h:
        sub = h.readlines()

    time_regex = re.compile(r'[0-9]{2}:[0-9]{2}:[0-9]{2},[0-9]{3} -->')
    activity_regex = re.compile(r'[a-zA-Z]')

    # Get start and end times
    times = list(filter(time_regex.search, sub))
    start_times = [time.split(' ')[0] for time in times]
    end_times = [time.split('--> ')[-1].replace('\n', '') for time in times]

    # Get activities
    acts = list(filter(activity_regex.search, sub))
    activities = [act.replace('\n', '') for act in acts]

    data['label'] = 'null'
    for s_time, e_time, act in zip(start_times, end_times, activities):
        start, end = time_to_hertz(s_time, e_time, sampling_rate)
        data.iloc[start:end, -1] = act

    return data


def time_to_hertz(start, end, rate):
    """
    Function which converts milliseconds to hertz timestamps

    :param start: float
        Start time in milliseconds
    :param end: float
        End time in milliseconds
    :param rate: int
        Employed sampling rate during recording
    :return int, int
        Start and end time in hertz
    """
    ref = datetime.strptime('00:00:00,000', '%H:%M:%S,%f').timestamp() * 1000.0
    dstart = np.abs(ref) - np.abs(datetime.strptime(start, '%H:%M:%S,%f').timestamp() * 1000.0)
    dend = np.abs(ref) - np.abs(datetime.strptime(end, '%H:%M:%S,%f').timestamp() * 1000.0)

    adjusted_rate = rate / 1000
    return int(np.floor(float(dstart) * adjusted_rate)), int(np.floor(float(dend) * adjusted_rate))
