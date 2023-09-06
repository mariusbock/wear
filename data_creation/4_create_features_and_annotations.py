import json
import os
import numpy as np
import pandas as pd

from utils import apply_sliding_window, label_dict, convert_labels_to_annotation_json

# define split
sbjs = [['sbj_0', 'sbj_1', 'sbj_2', 'sbj_3', 'sbj_4', 'sbj_5'], ['sbj_6', 'sbj_7', 'sbj_8', 'sbj_9', 'sbj_10', 'sbj_11'], ['sbj_12', 'sbj_13', 'sbj_14', 'sbj_15', 'sbj_16', 'sbj_17']]

# change these parameters
window_size = 50
window_overlap = 50
frames = 60
stride = 30

# change output folder
raw_inertial_folder = './data/wear/raw/inertial'
inertial_folder = './data/wear/processed/inertial_features/{}_frames_{}_stride'.format(frames, stride)
i3d_folder = './data/wear/processed/i3d_features/{}_frames_{}_stride'.format(frames, stride)
combined_folder = './data/wear/processed/combined_features/{}_frames_{}_stride'.format(frames, stride)
anno_folder = './data/wear/annotations'

# fixed dataset properties
nb_sbjs = 18
fps = 60
sampling_rate = 50

for i, split_sbjs in enumerate(sbjs):
    wear_annotations = {'version': 'Wear', 'database': {}, 'label_dict': label_dict}
    for sbj in split_sbjs:
        raw_inertial_sbj = pd.read_csv(os.path.join(raw_inertial_folder, sbj + '.csv'), index_col=None)        
        inertial_sbj = raw_inertial_sbj.replace({"label": label_dict}).fillna(-1).to_numpy()
        inertial_sbj[:, -1] += 1
        _, win_sbj, _ = apply_sliding_window(inertial_sbj, window_size, window_overlap)
        flipped_sbj = np.transpose(win_sbj[:, :, 1:], (0,2,1))
        flat_win_sbj = win_sbj.reshape(win_sbj.shape[0], -1)
        output_inertial = flipped_sbj.reshape(flipped_sbj.shape[0], -1)
        output_i3d = np.load(os.path.join(i3d_folder, sbj + '.npy'))
        try:
            output_combined = np.concatenate((output_inertial, output_i3d), axis=1)
        except ValueError:
            print('had to chop')
            output_combined = np.concatenate((output_inertial[:output_i3d.shape[0], :], output_i3d), axis=1)

        np.save(os.path.join(inertial_folder, sbj + '.npy'), output_inertial)
        np.save(os.path.join(combined_folder, sbj + '.npy'), output_combined)

    # create video annotations
    for j in range(nb_sbjs):
        curr_sbj = "sbj_" + str(j)
        raw_inertial_sbj_t = pd.read_csv(os.path.join(raw_inertial_folder, curr_sbj + '.csv'), index_col=None)
        duration_seconds = len(raw_inertial_sbj_t) / sampling_rate
        sbj_annos = convert_labels_to_annotation_json(raw_inertial_sbj_t.iloc[:, -1], sampling_rate, fps, label_dict)
        if curr_sbj in split_sbjs:
            train_test = 'Validation'
        else:
            train_test = 'Training'
        wear_annotations['database']['sbj_' + str(int(j))] = {
            'subset': train_test,
            'duration': duration_seconds,
            'fps': fps,
            'annotations': sbj_annos,
            } 
        with open(os.path.join(anno_folder, 'wear_split_' + str(int(i + 1)) +  '.json'), 'w') as outfile:
            outfile.write(json.dumps(wear_annotations, indent = 4))
        
